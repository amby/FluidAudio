import Accelerate
import Foundation
import OSLog

// Computes cosine distance between two embeddings. Returns infinity if embeddings are invalid.
public func cosineDist(_ a: [Float], _ b: [Float]) -> Float {
    guard a.count == b.count else {
        return .infinity
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
        return .infinity
    }
    
    // Cosine similarity = dot product / (magnitude1 * magnitude2)
    let similarity = dotProduct / (magnitudeA * magnitudeB)
    
    // Cosine distance = 1 - similarity
    return 1.0 - similarity
}

// Calculates element-wise sum of two given embeddings.
func sumEmbeddings(_ a: [Float], _ b: [Float]) -> [Float] {
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
func divEmbedding(_ a: [Float], _ b: Float) -> [Float] {
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

// Type of distance between embeddings/clusters.
public enum DistanceType {
    case min // Minimal distance.
    case max // Maximum distance.
}

// Represents distances between embeddings/clusters.
public struct ClusterDistances {
    // Type of distances.
    let type: DistanceType
    
    // Minimal distances between embeddings/clusters. Index of the first embedding is the same as
    // the index of corresponding distance in the arrays.
    var distances: [Float]
    
    // Indices of second embeddings/clusters.
    var otherIndices: [Int]
    
    // Updates current distance for specified pair of indices if given distance is less than current.
    // Returns true if updating took place, false otherwise.
    @discardableResult
    mutating func tryUpdating(index: Int, otherIndex: Int, distance: Float) -> Bool {
        switch type {
        case .min:
            if distance < distances[index] {
                distances[index] = distance
                otherIndices[index] = otherIndex
                return true
            }
            return false
        case .max:
            if distance > distances[index] {
                distances[index] = distance
                otherIndices[index] = otherIndex
                return true
            }
            return false
        }
    }
}

// Clusterizes given embeddings using Agglomerative Hierarchical Clustering. Also accepts and
// returns minimal cluster distances (to avoid unnecessary computations when processing streamed
// audio embeddings).
public func clusterize(maxDistance: Float, // Maximum distance threshold.
                       minClusterCount: Int = 0, // Minumum number of clusters to get.
                       embeddings: [[Float]],
                       minClusterDistances: ClusterDistances) -> (ClusterDistances, [Cluster]) {
    guard !embeddings.isEmpty else {
        return (minClusterDistances, [])
    }
    
    var sums = embeddings
    var centroids = embeddings
    var clusters: [[Int]] = (0..<embeddings.count).map { [$0] }
    var clusterCount: Int = clusters.count
    
    let resultMinClusterDistances = updateClusterDistances(embeddings: embeddings,
                                                           clusterDistances: minClusterDistances)
    
    var minClusterDistances = resultMinClusterDistances
    while true {
        let (uIntMinIndex, minDistance) = vDSP.indexOfMinimum(minClusterDistances.distances)
//        print("SEG", uIntMinIndex, minDistance, maxDistance, minClusterDistances, clusters)
//        print("MIN DISTANCE: \(minDistance), \(minClusterDistances.distances), \(sums), \(centroids), \(clusters)")
        if minDistance >= maxDistance {
            // Clusterization is complete.
//            print("SEG CHECK", updateClusterDistances(
//                embeddings: centroids,
//                clusterDistances: ClusterDistances(
//                    type: .min,
//                    distances: [],
//                    otherIndices: []),
//                removedEmbeddingIndices: minClusterDistances.distances.enumerated().filter { $0.element.isNaN }.map { $0.offset } ))
            break
        }
        let minIndex = Int(uIntMinIndex)
        
        // Merge clusters.
        let otherIndex = minClusterDistances.otherIndices[minIndex]
        
        clusters[minIndex].append(contentsOf: clusters[otherIndex])
        clusters[otherIndex] = []
        
        sums[minIndex] = sumEmbeddings(sums[minIndex], sums[otherIndex])
        centroids[minIndex] = divEmbedding(sums[minIndex], Float(clusters[minIndex].count))

        clusterCount -= 1
        if clusterCount == minClusterCount {
            // The requested minimum number of clusters is reached.
            break
        }

//        minClusterDistances.distances[otherIndex] = Float.nan
//        minClusterDistances.otherIndices[otherIndex] = -1 // Optional.
        
        minClusterDistances = updateClusterDistances(embeddings: centroids,
                                                     clusterDistances: minClusterDistances,
                                                     updatedEmbeddingIndices: [minIndex],
                                                     removedEmbeddingIndices: [otherIndex])
    }

    return (
        resultMinClusterDistances,
        clusters
            .enumerated()
            .filter { !$0.element.isEmpty }
            .map { Cluster(embeddingIndices: $0.element,
                           centroid: centroids[$0.offset]) }
    )
}

// Updates cluster distances for given embeddings.
func updateClusterDistances(embeddings: [[Float]],
                            clusterDistances: ClusterDistances,
                            updatedEmbeddingIndices: [Int] = [],
                            removedEmbeddingIndices: [Int] = []) -> ClusterDistances {
    let defaultDistance: Float = clusterDistances.type == .min ? .infinity : -.infinity
    let extraCount = embeddings.count - clusterDistances.distances.count
    var clusterDistances = extraCount > 0 ?
        ClusterDistances(
            type: clusterDistances.type,
            distances: clusterDistances.distances + .init(repeating: defaultDistance, count: extraCount),
            otherIndices: clusterDistances.otherIndices + .init(repeating: -1, count: extraCount)
        ) : clusterDistances
    
    // Mark distances for removed embeddings as noop.
    for i in removedEmbeddingIndices {
        clusterDistances.distances[i] = .nan
        clusterDistances.otherIndices[i] = -1 // Optional.
    }
    
    // Compute distances for new embeddings if any.
    for i in embeddings.count-extraCount..<embeddings.count {
        for k in 0..<i {
            if clusterDistances.distances[k].isNaN {
                continue
            }
            let distance = cosineDist(embeddings[i], embeddings[k])
            clusterDistances.tryUpdating(index: i, otherIndex: k, distance: distance)
            clusterDistances.tryUpdating(index: k, otherIndex: i, distance: distance)
        }
    }

    var wrongDistanceIndices: [Int] = []

    // Reset distances for all removed embeddings.
    for i in 0..<embeddings.count {
        if clusterDistances.distances[i].isNaN {
            continue
        }
        if removedEmbeddingIndices.contains(clusterDistances.otherIndices[i]) {
            clusterDistances.distances[i] = defaultDistance
            clusterDistances.otherIndices[i] = -1
            wrongDistanceIndices.append(i)
        }
    }
    
    // Recompute min distances for updated embeddings.
    for i in updatedEmbeddingIndices {
        for k in 0..<embeddings.count {
            if i == k || clusterDistances.distances[k].isNaN {
                continue
            }
            let distance = cosineDist(embeddings[i], embeddings[k])
            if clusterDistances.tryUpdating(index: k, otherIndex: i, distance: distance) {
                clusterDistances.tryUpdating(index: i, otherIndex: k, distance: distance)
            } else if clusterDistances.otherIndices[k] == i {
                // Previous minimal distance was computed with for the previous i-th embedding.
                // Should be recomputed since i-th embedding was changed.
                wrongDistanceIndices.append(k)
                clusterDistances.distances[k] = defaultDistance
                clusterDistances.otherIndices[k] = -1
            }
        }
    }
    
    // Recomputed distances which were invalidated by updating of embeddings.
    for i in wrongDistanceIndices {
        for k in 0..<embeddings.count {
            if i == k || clusterDistances.distances[k].isNaN {
                continue
            }
            let distance = cosineDist(embeddings[i], embeddings[k])
            clusterDistances.tryUpdating(index: i, otherIndex: k, distance: distance)
//            clusterDistances.tryUpdating(index: k, otherIndex: i, distance: distance)
        }
    }
    
    return clusterDistances
}

// Computes distance from given embeddings to specified clusters.
func computeDistancesToClusters(embeddings: [[Float]],
                                clusters: [Cluster],
                                distancesToClusters: ClusterDistances,
                                removedEmbeddingIndex: Int? = nil,
                                removedClusterIndex: Int? = nil)  -> ClusterDistances {
    guard !clusters.isEmpty else {
        return ClusterDistances(type: distancesToClusters.type, distances: [], otherIndices: [])
    }
    
    if let removedClusterIndex {
        precondition(clusters[removedClusterIndex].centroid.isEmpty,
                     "Found filtered out cluster with non-empty centroid")
    }

    let defaultDistance: Float = distancesToClusters.type == .min ? .infinity : -.infinity
    let extraCount = embeddings.count - distancesToClusters.distances.count
    var distancesToClusters = extraCount > 0 ?
        ClusterDistances(
            type: distancesToClusters.type,
            distances: distancesToClusters.distances + .init(repeating: defaultDistance, count: extraCount),
            otherIndices: distancesToClusters.otherIndices + .init(repeating: -1, count: extraCount)
        ) : distancesToClusters

    if let removedEmbeddingIndex = removedEmbeddingIndex {
        distancesToClusters.distances[removedEmbeddingIndex] = Float.nan
    }
    
    for i in 0..<embeddings.count {
        if distancesToClusters.distances[i].isNaN {
            continue
        }
        if let removedClusterIndex, distancesToClusters.otherIndices[i] == removedClusterIndex {
            distancesToClusters.distances[i] = defaultDistance
            distancesToClusters.otherIndices[i] = -1
        }
        let embedding = embeddings[i]
        for (k, cluster) in clusters.enumerated() {
            if cluster.centroid.isEmpty {
                // Filtered out cluster.
                continue
            }
            let distance = cosineDist(embedding, cluster.centroid)
            distancesToClusters.tryUpdating(index: i, otherIndex: k, distance: distance)
        }
    }
    
    return distancesToClusters
}
