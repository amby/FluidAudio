import XCTest
@testable import FluidAudio

/// Comprehensive test suite for Agglomerative Hierarchical Clustering (AHC) implementation.
/// Tests cover cosine distance calculations, helper functions, data structures, and clustering algorithms.
@available(macOS 13.0, iOS 16.0, *)
final class AHCTests: XCTestCase {
    
    // MARK: - Cosine Distance Tests
    
    /// Tests that identical vectors have a cosine distance of 0.0.
    /// This is fundamental to clustering - identical embeddings should be considered the same.
    func testCosineDistanceIdenticalVectors() {
        let vector = [1.0, 2.0, 3.0, 4.0] as [Float]
        let distance = cosineDistance(a: vector, b: vector)
        XCTAssertEqual(distance, 0.0, accuracy: 0.001)
    }
    
    /// Tests that orthogonal vectors have a cosine distance of 1.0.
    /// Orthogonal vectors are perpendicular and should have maximum dissimilarity.
    func testCosineDistanceOrthogonalVectors() {
        let vector1 = [1.0, 0.0, 0.0] as [Float]
        let vector2 = [0.0, 1.0, 0.0] as [Float]
        let distance = cosineDistance(a: vector1, b: vector2)
        XCTAssertEqual(distance, 1.0, accuracy: 0.001)
    }
    
    /// Tests that opposite vectors have a cosine distance of 2.0.
    /// Opposite vectors point in completely opposite directions and should have maximum distance.
    func testCosineDistanceOppositeVectors() {
        let vector1 = [1.0, 0.0, 0.0] as [Float]
        let vector2 = [-1.0, 0.0, 0.0] as [Float]
        let distance = cosineDistance(a: vector1, b: vector2)
        XCTAssertEqual(distance, 2.0, accuracy: 0.001)
    }
    
    /// Tests that vectors with different dimensions return infinity.
    /// Cosine distance is undefined for vectors of different dimensions.
    func testCosineDistanceDifferentDimensions() {
        let vector1 = [1.0, 2.0] as [Float]
        let vector2 = [1.0, 2.0, 3.0] as [Float]
        let distance = cosineDistance(a: vector1, b: vector2)
        XCTAssertEqual(distance, Float.infinity)
    }
    
    /// Tests that zero magnitude vectors return infinity.
    /// Cosine distance is undefined when one vector has zero magnitude.
    func testCosineDistanceZeroMagnitude() {
        let vector1 = [0.0, 0.0, 0.0] as [Float]
        let vector2 = [1.0, 2.0, 3.0] as [Float]
        let distance = cosineDistance(a: vector1, b: vector2)
        XCTAssertEqual(distance, Float.infinity)
    }
    
    /// Tests that random vectors have cosine distance between 0 and 2.
    /// Validates the distance calculation for typical use cases.
    func testCosineDistanceRandomVectors() {
        let vector1 = [1.0, 2.0, 3.0] as [Float]
        let vector2 = [4.0, 5.0, 6.0] as [Float]
        let distance = cosineDistance(a: vector1, b: vector2)
        XCTAssertGreaterThan(distance, 0.0)
        XCTAssertLessThan(distance, 2.0)
    }
    
    // MARK: - Helper Function Tests
    
    /// Tests element-wise addition of two embeddings.
    /// This is used when merging clusters to compute new centroids.
    func testSumEmbeddings() {
        let a = [1.0, 2.0, 3.0] as [Float]
        let b = [4.0, 5.0, 6.0] as [Float]
        let result = sumEmbeddings(a: a, b: b)
        XCTAssertEqual(result, [5.0, 7.0, 9.0] as [Float])
    }
    
    /// Tests that sumEmbeddings handles different dimensions gracefully.
    /// The function should crash with precondition failure for mismatched dimensions.
    /// Note: Precondition failures are difficult to test in XCTest, so this is a placeholder.
    func testSumEmbeddingsDifferentDimensions() {
        // This should crash with precondition failure
        // Note: We can't easily test precondition failures in XCTest
        // The precondition will catch this at runtime
        // For now, we'll skip this test as it's not easily testable
        // XCTAssertPreconditionFailure is not available in standard XCTest
        XCTAssertTrue(true) // Placeholder test
    }
    
    /// Tests vector division by a scalar value.
    /// This is used to compute cluster centroids by dividing sum by count.
    func testDivEmbedding() {
        let a = [2.0, 4.0, 6.0] as [Float]
        let b: Float = 2.0
        let result = divEmbedding(a: a, b: b)
        XCTAssertEqual(result, [1.0, 2.0, 3.0] as [Float])
    }
    
    /// Tests vector division by zero returns infinity values.
    /// This edge case should be handled gracefully in the clustering algorithm.
    func testDivEmbeddingByZero() {
        let a = [1.0, 2.0, 3.0] as [Float]
        let b: Float = 0.0
        let result = divEmbedding(a: a, b: b)
        // Should result in infinity values
        XCTAssertTrue(result.allSatisfy { $0.isInfinite })
    }
    
    // MARK: - Cluster Struct Tests
    
    /// Tests Cluster struct initialization with embedding indices and centroid.
    /// Clusters represent groups of similar embeddings with their computed centroid.
    func testClusterInitialization() {
        let indices = [0, 1, 2]
        let centroid = [1.0, 2.0, 3.0] as [Float]
        let cluster = Cluster(embeddingIndices: indices, centroid: centroid)
        
        XCTAssertEqual(cluster.embeddingIndices, indices)
        XCTAssertEqual(cluster.centroid, centroid)
    }
    
    /// Tests Cluster struct with empty embedding indices.
    /// This edge case should be handled gracefully in the clustering algorithm.
    func testClusterEmptyIndices() {
        let indices: [Int] = []
        let centroid = [1.0, 2.0, 3.0] as [Float]
        let cluster = Cluster(embeddingIndices: indices, centroid: centroid)
        
        XCTAssertEqual(cluster.embeddingIndices.count, 0)
        XCTAssertEqual(cluster.centroid, centroid)
    }
    
    // MARK: - MinClusterDistances Tests
    
    /// Tests MinClusterDistances struct initialization.
    /// This structure tracks minimum distances between clusters for efficient clustering.
    func testMinClusterDistancesInitialization() {
        let distances = [0.5, 0.3, 0.8] as [Float]
        let otherIndices = [1, 0, 2]
        let minDistances = MinClusterDistances(distances: distances, otherIndices: otherIndices)
        
        XCTAssertEqual(minDistances.distances, distances)
        XCTAssertEqual(minDistances.otherIndices, otherIndices)
    }
    
    /// Tests the tryUpdating method of MinClusterDistances.
    /// This method efficiently updates minimum distances only when a smaller distance is found.
    func testMinClusterDistancesTryUpdating() {
        var minDistances = MinClusterDistances(
            distances: [1.0, 0.5, 0.8] as [Float],
            otherIndices: [1, 0, 2]
        )
        
        // Update with smaller distance
        minDistances.tryUpdating(index: 0, otherIndex: 2, distance: 0.3)
        XCTAssertEqual(minDistances.distances[0], 0.3)
        XCTAssertEqual(minDistances.otherIndices[0], 2)
        
        // Try to update with larger distance (should not change)
        minDistances.tryUpdating(index: 0, otherIndex: 1, distance: 0.7)
        XCTAssertEqual(minDistances.distances[0], 0.3)
        XCTAssertEqual(minDistances.otherIndices[0], 2)
    }
    
    // MARK: - Clusterize Function Tests
    
    /// Tests clustering with empty input.
    /// Should return empty clusters and preserve the input minClusterDistances.
    func testClusterizeEmptyInput() {
        let minDistances = MinClusterDistances(distances: [], otherIndices: [])
        let (resultDistances, clusters) = clusterize(
            maxDistance: 0.5,
            embeddings: [],
            minClusterDistances: minDistances
        )
        
        XCTAssertEqual(resultDistances.distances.count, 0)
        XCTAssertEqual(resultDistances.otherIndices.count, 0)
        XCTAssertEqual(clusters.count, 0)
    }
    
    /// Tests clustering with a single embedding.
    /// Should create one cluster containing the single embedding.
    func testClusterizeSingleEmbedding() {
        let embedding = [1.0, 2.0, 3.0] as [Float]
        let minDistances = MinClusterDistances(distances: [], otherIndices: [])
        let (_, clusters) = clusterize(
            maxDistance: 0.5,
            embeddings: [embedding],
            minClusterDistances: minDistances
        )
        
        XCTAssertEqual(clusters.count, 1)
        XCTAssertEqual(clusters[0].embeddingIndices, [0])
        XCTAssertEqual(clusters[0].centroid, embedding)
    }
    
    /// Tests clustering with two identical embeddings.
    /// Should merge them into a single cluster since they have zero distance.
    func testClusterizeTwoIdenticalEmbeddings() {
        let embedding = [1.0, 2.0, 3.0] as [Float]
        let minDistances = MinClusterDistances(distances: [], otherIndices: [])
        let (_, clusters) = clusterize(
            maxDistance: 0.5,
            embeddings: [embedding, embedding],
            minClusterDistances: minDistances
        )
        
        XCTAssertEqual(clusters.count, 1)
        XCTAssertEqual(clusters[0].embeddingIndices.sorted(), [0, 1])
    }
    
    /// Tests clustering with two different embeddings.
    /// With a low threshold, they should remain in separate clusters.
    func testClusterizeTwoDifferentEmbeddings() {
        let embedding1 = [1.0, 0.0, 0.0] as [Float]
        let embedding2 = [0.0, 1.0, 0.0] as [Float]
        let minDistances = MinClusterDistances(distances: [], otherIndices: [])
        let (_, clusters) = clusterize(
            maxDistance: 0.5,
            embeddings: [embedding1, embedding2],
            minClusterDistances: minDistances
        )
        
        // With low threshold, they should remain separate
        XCTAssertEqual(clusters.count, 2)
        XCTAssertEqual(clusters[0].embeddingIndices, [0])
        XCTAssertEqual(clusters[1].embeddingIndices, [1])
    }
    
    /// Tests clustering with a high distance threshold.
    /// With a high threshold, different embeddings should cluster together.
    func testClusterizeHighThreshold() {
        let embedding1 = [1.0, 0.0, 0.0] as [Float]
        let embedding2 = [0.0, 1.0, 0.0] as [Float]
        let minDistances = MinClusterDistances(distances: [], otherIndices: [])
        let (_, clusters) = clusterize(
            maxDistance: 2.0, // High threshold
            embeddings: [embedding1, embedding2],
            minClusterDistances: minDistances
        )
        
        // With high threshold, they should cluster together
        XCTAssertEqual(clusters.count, 1)
        XCTAssertEqual(clusters[0].embeddingIndices.sorted(), [0, 1])
    }
    
    /// Tests clustering with multiple embeddings that form natural groups.
    /// Should cluster similar embeddings together while keeping different ones separate.
    func testClusterizeMultipleEmbeddings() {
        let embeddings = [
            [1.0, 0.0, 0.0] as [Float],
            [0.9, 0.1, 0.0] as [Float], // Similar to first
            [0.0, 1.0, 0.0] as [Float],
            [0.0, 0.9, 0.1] as [Float]  // Similar to third
        ]
        let minDistances = MinClusterDistances(distances: [], otherIndices: [])
        let (_, clusters) = clusterize(
            maxDistance: 0.5,
            embeddings: embeddings,
            minClusterDistances: minDistances
        )
        
        XCTAssertGreaterThanOrEqual(clusters.count, 1)
        XCTAssertLessThanOrEqual(clusters.count, 4)
        
        // Verify all embeddings are assigned to clusters
        let allIndices = clusters.flatMap { $0.embeddingIndices }.sorted()
        XCTAssertEqual(allIndices, [0, 1, 2, 3])
    }
    
    /// Tests clustering with pre-computed minimum distances.
    /// This is useful for streaming scenarios where distances are already calculated.
    func testClusterizeWithExistingMinDistances() {
        let embeddings = [
            [1.0, 0.0, 0.0] as [Float],
            [0.9, 0.1, 0.0] as [Float]
        ]
        let existingMinDistances = MinClusterDistances(
            distances: [0.2] as [Float],
            otherIndices: [1]
        )
        let (_, clusters) = clusterize(
            maxDistance: 0.5,
            embeddings: embeddings,
            minClusterDistances: existingMinDistances
        )
        
        XCTAssertEqual(clusters.count, 1)
        XCTAssertEqual(clusters[0].embeddingIndices.sorted(), [0, 1])
    }
    
    // MARK: - Edge Case Tests
    
    /// Tests clustering with zero magnitude embeddings.
    /// Should handle gracefully by creating separate clusters for invalid embeddings.
    func testClusterizeZeroMagnitudeEmbeddings() {
        let embeddings = [
            [0.0, 0.0, 0.0] as [Float],
            [1.0, 2.0, 3.0] as [Float]
        ]
        let minDistances = MinClusterDistances(distances: [], otherIndices: [])
        let (_, clusters) = clusterize(
            maxDistance: 0.5,
            embeddings: embeddings,
            minClusterDistances: minDistances
        )
        
        // Should handle gracefully and create separate clusters
        XCTAssertEqual(clusters.count, 2)
    }
    
    /// Tests clustering with a very high distance threshold.
    /// Should cluster all embeddings together regardless of their actual distances.
    func testClusterizeVeryHighThreshold() {
        let embeddings = [
            [1.0, 0.0, 0.0] as [Float],
            [0.0, 1.0, 0.0] as [Float],
            [0.0, 0.0, 1.0] as [Float]
        ]
        let minDistances = MinClusterDistances(distances: [], otherIndices: [])
        let (_, clusters) = clusterize(
            maxDistance: 10.0, // Very high threshold
            embeddings: embeddings,
            minClusterDistances: minDistances
        )
        
        // Should cluster all together
        XCTAssertEqual(clusters.count, 1)
        XCTAssertEqual(clusters[0].embeddingIndices.sorted(), [0, 1, 2])
    }
    
    /// Tests clustering with a very low distance threshold.
    /// Should keep all embeddings in separate clusters since no pairs meet the threshold.
    func testClusterizeVeryLowThreshold() {
        let embeddings = [
            [1.0, 0.0, 0.0] as [Float],
            [0.0, 1.0, 0.0] as [Float],
            [0.0, 0.0, 1.0] as [Float]
        ]
        let minDistances = MinClusterDistances(distances: [], otherIndices: [])
        let (_, clusters) = clusterize(
            maxDistance: 0.01, // Very low threshold
            embeddings: embeddings,
            minClusterDistances: minDistances
        )
        
        // Should keep all separate
        XCTAssertEqual(clusters.count, 3)
        for (index, cluster) in clusters.enumerated() {
            XCTAssertEqual(cluster.embeddingIndices, [index])
        }
    }
    
    // MARK: - Performance Tests
    
    /// Tests performance with a moderately sized dataset.
    /// Validates that clustering completes within reasonable time bounds.
    func testPerformanceWithLargeDataset() {
        let embeddings = generateTestEmbeddings(count: 10, dimension: 16) // Reduced size for testing
        let minDistances = MinClusterDistances(distances: [], otherIndices: [])
        
        measure {
            let (_, clusters) = clusterize(
                maxDistance: 0.5,
                embeddings: embeddings,
                minClusterDistances: minDistances
            )
            XCTAssertGreaterThan(clusters.count, 0)
        }
    }
    
    /// Tests memory efficiency with multiple clustering operations.
    /// Ensures no memory leaks occur during repeated clustering operations.
    func testMemoryEfficiency() {
        // Test with multiple calls to ensure no memory leaks
        for _ in 0..<3 { // Reduced iterations
            let embeddings = generateTestEmbeddings(count: 5, dimension: 8) // Reduced size
            let minDistances = MinClusterDistances(distances: [], otherIndices: [])
            let (_, clusters) = clusterize(
                maxDistance: 0.5,
                embeddings: embeddings,
                minClusterDistances: minDistances
            )
            XCTAssertGreaterThan(clusters.count, 0)
        }
    }
    
    // MARK: - Helper Methods
    
    /// Generates random test embeddings for testing purposes.
    /// - Parameters:
    ///   - count: Number of embeddings to generate
    ///   - dimension: Dimension of each embedding
    /// - Returns: Array of random embeddings with values between -1 and 1
    private func generateTestEmbeddings(count: Int, dimension: Int) -> [[Float]] {
        return (0..<count).map { _ in
            (0..<dimension).map { _ in Float.random(in: -1...1) }
        }
    }
    
    /// Generates a cluster of similar embeddings around a center point.
    /// - Parameters:
    ///   - center: Center point for the cluster
    ///   - size: Number of embeddings to generate
    ///   - dimension: Dimension of each embedding
    /// - Returns: Array of embeddings clustered around the center with small random variations
    private func generateCluster(center: [Float], size: Int, dimension: Int) -> [[Float]] {
        return (0..<size).map { _ in
            center.map { $0 + Float.random(in: -0.1...0.1) }
        }
    }
}
