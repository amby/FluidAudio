import Accelerate
import Foundation
import OSLog

// Computes cosine distance between two embeddings. Returns infinity if embeddings are invalid.
public func cosineDistance(a: [Float], b: [Float]) -> Float {
    guard a.count == b.count else {
        return Float.infinity
    }
    
    let dimension = a.count
    
    var dotProduct: Float = 0
    vDSP_dotpr(a, 1, b, 1, &dotProduct, vDSP_Length(dimension))
    
    var magnitudeA: Float = 0
    var magnitudeB: Float = 0
    vDSP_svesq(a, 1, &magnitudeA, vDSP_Length(dimension))
    vDSP_svesq(b, 1, &magnitudeB, vDSP_Length(dimension))
    
    magnitudeA = sqrt(magnitudeA)
    magnitudeB = sqrt(magnitudeB)
    
    guard magnitudeA > 0 && magnitudeB > 0 else {
        return Float.infinity
    }
    
    // Cosine similarity = dot product / (magnitude1 * magnitude2)
    let similarity = dotProduct / (magnitudeA * magnitudeB)
    
    // Cosine distance = 1 - similarity
    return 1.0 - similarity
}

// Calculates element-wise sum of two given embeddings.
func sumEmbeddings(a: [Float], b: [Float]) -> [Float] {
    precondition(a.count == b.count, "Different embedding dimensions")
    let count = a.count
    let c = [Float](unsafeUninitializedCapacity: count) {
        buffer, initializedCount in
                
        vDSP_vadd(a, 1,
                  b, 1,
                  buffer.baseAddress!, 1,
                  vDSP_Length(count))
        
        initializedCount = count
    }
    return c
}

// Divides given vector by specified scalar.
func divEmbedding(a: [Float], b: Float) -> [Float] {
    let count = a.count
    let c = [Float](unsafeUninitializedCapacity: count) {
        buffer, initializedCount in
        
        vDSP_vsdiv(a, 1,
                   [b],
                   buffer.baseAddress!, 1,
                   vDSP_Length(count))
        
        initializedCount = count
    }
    return c
}

// Embedding cluster.
public struct Cluster {
    // Indices of embeddings included into the cluster.
    let embeddingIndices: [Int]

    // Centroid of the embeddings included into the cluster.
    let centroid: [Float]
}

// Represents minimal distances between embeddings/clusters.
public struct MinClusterDistances {
    // Minimal distances between embeddings/clusters. Index of the first embedding is the same as
    // the index of corresponding distance in the arrays.
    var distances: [Float]
    
    // Indices of second embeddings/clusters.
    var otherIndices: [Int]
    
    // Updates current distance for specified pair of indices if given distance is less than current.
    mutating func tryUpdating(index: Int, otherIndex: Int, distance: Float) {
        if distance < distances[index] {
            distances[index] = distance
            otherIndices[index] = otherIndex
        }
    }
}

// Clusterizes given embeddings using Agglomerative Hierarchical Clustering. Also accepts and
// returns minimal cluster distances (to avoid unnecessary computations when processing streamed
// audio embeddings).
public func clusterize(maxDistance: Float, // Maximum distance threshold.
                       embeddings: [[Float]],
                       minClusterDistances: MinClusterDistances) -> (MinClusterDistances, [Cluster]) {
    guard !embeddings.isEmpty else {
        return (minClusterDistances, [])
    }
    
    var sums = embeddings
    var centroids = embeddings
    var clusters: [[Int]] = (0..<embeddings.count).map { [$0] }
    
    let resultMinClusterDistances = updateMinClusterDistances(embeddings: embeddings,
                                                              minClusterDistances: minClusterDistances)
    
    var minClusterDistances = resultMinClusterDistances
    while true {
        let (uIntMinIndex, minDistance) = vDSP.indexOfMinimum(minClusterDistances.distances)
//        print("MIN DISTANCE: \(minDistance), \(minClusterDistances.distances), \(sums), \(centroids), \(clusters)")
        if minDistance >= maxDistance {
            // Clusterization is complete.
            break
        }
        let minIndex = Int(uIntMinIndex)
        
        // Merge clusters.
        let otherIndex = minClusterDistances.otherIndices[minIndex]
        
        clusters[minIndex].append(contentsOf: clusters[otherIndex])
        clusters[otherIndex] = []
        
        sums[minIndex] = sumEmbeddings(a: sums[minIndex], b: sums[otherIndex])
        centroids[minIndex] = divEmbedding(a: sums[minIndex], b: Float(clusters[minIndex].count))
        
//        minClusterDistances.distances[otherIndex] = Float.nan
//        minClusterDistances.otherIndices[otherIndex] = -1 // Optional.
        
        minClusterDistances = updateMinClusterDistances(embeddings: centroids,
                                                        minClusterDistances: minClusterDistances,
                                                        updatedEmbeddingIndices: [minIndex],
                                                        removedEmbeddingIndices: [otherIndex])
    }

    return (
        resultMinClusterDistances,
        clusters
            .filter { !$0.isEmpty }
            .enumerated().map { Cluster(embeddingIndices: $0.element,
                                        centroid: centroids[$0.offset]) }
    )
}

// Updates minimal cluster distances for given embeddings.
func updateMinClusterDistances(embeddings: [[Float]],
                               minClusterDistances: MinClusterDistances,
                               updatedEmbeddingIndices: [Int] = [],
                               removedEmbeddingIndices: [Int] = []) -> MinClusterDistances {
    let extraCount = embeddings.count - minClusterDistances.distances.count
    var minClusterDistances = extraCount > 0 ?
        MinClusterDistances(
            distances: minClusterDistances.distances + .init(repeating: Float.infinity, count: extraCount),
            otherIndices: minClusterDistances.otherIndices + .init(repeating: -1, count: extraCount)
        ) : minClusterDistances
    
    // Mark distances for removed embeddings as noop.
    for i in removedEmbeddingIndices {
        minClusterDistances.distances[i] = .nan
        minClusterDistances.otherIndices[i] = -1 // Optional.
    }
    
    // Compute distances for new embeddings if any.
    for i in embeddings.count-extraCount..<embeddings.count {
        for k in 0..<i {
            if minClusterDistances.distances[k].isNaN {
                continue
            }
            let distance = cosineDistance(a: embeddings[i], b: embeddings[k])
            minClusterDistances.tryUpdating(index: i, otherIndex: k, distance: distance)
            minClusterDistances.tryUpdating(index: k, otherIndex: i, distance: distance)
        }
    }

    var wrongDistanceIndices: [Int] = []

    // Reset distances for all removed embeddings.
    for i in 0..<embeddings.count {
        if minClusterDistances.distances[i].isNaN {
            continue
        }
        if removedEmbeddingIndices.contains(minClusterDistances.otherIndices[i]) {
            minClusterDistances.distances[i] = .infinity
            minClusterDistances.otherIndices[i] = -1
            wrongDistanceIndices.append(i)
        }
    }
    
    // Recompute min distances for updated embeddings.
    for i in updatedEmbeddingIndices {
        for k in 0..<embeddings.count {
            if i == k || minClusterDistances.distances[k].isNaN {
                continue
            }
            let distance = cosineDistance(a: embeddings[i], b: embeddings[k])
            if distance < minClusterDistances.distances[k] {
                minClusterDistances.tryUpdating(index: i, otherIndex: k, distance: distance)
                minClusterDistances.tryUpdating(index: k, otherIndex: i, distance: distance)
            } else if minClusterDistances.otherIndices[k] == i {
                // Previous minimal distance was computed with for the previous i-th embedding.
                // Should be recomputed since i-th embedding was changed.
                wrongDistanceIndices.append(k)
                minClusterDistances.distances[k] = .infinity
                minClusterDistances.otherIndices[k] = -1
            }
        }
    }
    
    // Recomputed distances which were invalidated by updating of embeddings.
    for i in wrongDistanceIndices {
        for k in 0..<embeddings.count {
            if i == k || minClusterDistances.distances[k].isNaN {
                continue
            }
            let distance = cosineDistance(a: embeddings[i], b: embeddings[k])
            minClusterDistances.tryUpdating(index: i, otherIndex: k, distance: distance)
//            minClusterDistances.tryUpdating(index: k, otherIndex: i, distance: distance)
        }
    }
    
    return minClusterDistances
}
