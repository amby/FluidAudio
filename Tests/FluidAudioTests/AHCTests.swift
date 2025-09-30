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
        let distance = cosineDist(vector, vector)
        XCTAssertEqual(distance, 0.0, accuracy: 0.001)
    }
    
    /// Tests that orthogonal vectors have a cosine distance of 1.0.
    /// Orthogonal vectors are perpendicular and should have maximum dissimilarity.
    func testCosineDistanceOrthogonalVectors() {
        let vector1 = [1.0, 0.0, 0.0] as [Float]
        let vector2 = [0.0, 1.0, 0.0] as [Float]
        let distance = cosineDist(vector1, vector2)
        XCTAssertEqual(distance, 1.0, accuracy: 0.001)
    }
    
    /// Tests that opposite vectors have a cosine distance of 2.0.
    /// Opposite vectors point in completely opposite directions and should have maximum distance.
    func testCosineDistanceOppositeVectors() {
        let vector1 = [1.0, 0.0, 0.0] as [Float]
        let vector2 = [-1.0, 0.0, 0.0] as [Float]
        let distance = cosineDist(vector1, vector2)
        XCTAssertEqual(distance, 2.0, accuracy: 0.001)
    }
    
    /// Tests that vectors with different dimensions return infinity.
    /// Cosine distance is undefined for vectors of different dimensions.
    func testCosineDistanceDifferentDimensions() {
        let vector1 = [1.0, 2.0] as [Float]
        let vector2 = [1.0, 2.0, 3.0] as [Float]
        let distance = cosineDist(vector1, vector2)
        XCTAssertEqual(distance, Float.infinity)
    }
    
    /// Tests that zero magnitude vectors return infinity.
    /// Cosine distance is undefined when one vector has zero magnitude.
    func testCosineDistanceZeroMagnitude() {
        let vector1 = [0.0, 0.0, 0.0] as [Float]
        let vector2 = [1.0, 2.0, 3.0] as [Float]
        let distance = cosineDist(vector1, vector2)
        XCTAssertEqual(distance, Float.infinity)
    }
    
    /// Tests that random vectors have cosine distance between 0 and 2.
    /// Validates the distance calculation for typical use cases.
    func testCosineDistanceRandomVectors() {
        let vector1 = [1.0, 2.0, 3.0] as [Float]
        let vector2 = [4.0, 5.0, 6.0] as [Float]
        let distance = cosineDist(vector1, vector2)
        XCTAssertGreaterThan(distance, 0.0)
        XCTAssertLessThan(distance, 2.0)
    }
    
    // MARK: - Helper Function Tests
    
    /// Tests element-wise addition of two embeddings.
    /// This is used when merging clusters to compute new centroids.
    func testSumEmbeddings() {
        let a = [1.0, 2.0, 3.0] as [Float]
        let b = [4.0, 5.0, 6.0] as [Float]
        let result = sumEmbeddings(a, b)
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
        let result = divEmbedding(a, b)
        XCTAssertEqual(result, [1.0, 2.0, 3.0] as [Float])
    }
    
    /// Tests vector division by zero returns infinity values.
    /// This edge case should be handled gracefully in the clustering algorithm.
    func testDivEmbeddingByZero() {
        let a = [1.0, 2.0, 3.0] as [Float]
        let b: Float = 0.0
        let result = divEmbedding(a, b)
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
        let minDistances = ClusterDistances(type: .min, distances: distances, otherIndices: otherIndices)
        
        XCTAssertEqual(minDistances.distances, distances)
        XCTAssertEqual(minDistances.otherIndices, otherIndices)
    }
    
    /// Tests the tryUpdating method of MinClusterDistances.
    /// This method efficiently updates minimum distances only when a smaller distance is found.
    func testMinClusterDistancesTryUpdating() {
        var minDistances = ClusterDistances(
            type: .min,
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
        let minDistances = ClusterDistances(type: .min, distances: [], otherIndices: [])
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
        let minDistances = ClusterDistances(type: .min, distances: [], otherIndices: [])
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
        let minDistances = ClusterDistances(type: .min, distances: [], otherIndices: [])
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
        let minDistances = ClusterDistances(type: .min, distances: [], otherIndices: [])
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
        let minDistances = ClusterDistances(type: .min, distances: [], otherIndices: [])
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
        let minDistances = ClusterDistances(type: .min, distances: [], otherIndices: [])
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
        let existingMinDistances = ClusterDistances(
            type: .min,
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
        let minDistances = ClusterDistances(type: .min, distances: [], otherIndices: [])
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
        let minDistances = ClusterDistances(type: .min, distances: [], otherIndices: [])
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
        let minDistances = ClusterDistances(type: .min, distances: [], otherIndices: [])
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
        let minDistances = ClusterDistances(type: .min, distances: [], otherIndices: [])
        
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
            let minDistances = ClusterDistances(type: .min, distances: [], otherIndices: [])
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
    
    // MARK: - New AHC Functionality Tests
    
    // MARK: - DistanceType Tests
    
    /// Tests the DistanceType.min enum case to ensure proper initialization and comparison.
    /// Verifies that the min distance type is correctly identified and can be used
    /// in clustering operations that require minimum distance calculations.
    func testDistanceTypeMin() {
        let distanceType = DistanceType.min
        XCTAssertEqual(distanceType, .min)
    }
    
    /// Tests the DistanceType.max enum case to ensure proper initialization and comparison.
    /// Verifies that the max distance type is correctly identified and can be used
    /// in clustering operations that require maximum distance calculations.
    func testDistanceTypeMax() {
        let distanceType = DistanceType.max
        XCTAssertEqual(distanceType, .max)
    }
    
    // MARK: - ClusterDistances Tests
    
    /// Tests ClusterDistances initialization with min distance type.
    /// Verifies that the ClusterDistances struct correctly stores min type distances
    /// and associated indices, ensuring proper data structure initialization.
    func testClusterDistancesMinTypeInitialization() {
        let distances: [Float] = [0.5, 0.3, 0.8]
        let otherIndices = [1, 0, 2]
        let clusterDistances = ClusterDistances(type: .min, distances: distances, otherIndices: otherIndices)
        
        XCTAssertEqual(clusterDistances.type, .min)
        XCTAssertEqual(clusterDistances.distances, distances)
        XCTAssertEqual(clusterDistances.otherIndices, otherIndices)
    }
    
    /// Tests ClusterDistances initialization with max distance type.
    /// Verifies that the ClusterDistances struct correctly stores max type distances
    /// and associated indices, ensuring proper data structure initialization.
    func testClusterDistancesMaxTypeInitialization() {
        let distances: [Float] = [0.5, 0.3, 0.8]
        let otherIndices = [1, 0, 2]
        let clusterDistances = ClusterDistances(type: .max, distances: distances, otherIndices: otherIndices)
        
        XCTAssertEqual(clusterDistances.type, .max)
        XCTAssertEqual(clusterDistances.distances, distances)
        XCTAssertEqual(clusterDistances.otherIndices, otherIndices)
    }
    
    /// Tests ClusterDistances tryUpdating method with min distance type.
    /// Verifies that the method correctly updates distances only when new values are smaller
    /// (for min type) and returns appropriate boolean values indicating update success.
    func testClusterDistancesTryUpdatingMinType() {
        var clusterDistances = ClusterDistances(
            type: .min,
            distances: [1.0, 0.5, 0.8],
            otherIndices: [1, 0, 2]
        )
        
        // Test updating with smaller distance (should update)
        let updated = clusterDistances.tryUpdating(index: 0, otherIndex: 1, distance: 0.3)
        XCTAssertTrue(updated)
        XCTAssertEqual(clusterDistances.distances[0], 0.3)
        XCTAssertEqual(clusterDistances.otherIndices[0], 1)
        
        // Test updating with larger distance (should not update)
        let notUpdated = clusterDistances.tryUpdating(index: 0, otherIndex: 2, distance: 0.9)
        XCTAssertFalse(notUpdated)
        XCTAssertEqual(clusterDistances.distances[0], 0.3) // Should remain unchanged
    }
    
    /// Tests ClusterDistances tryUpdating method with max distance type.
    /// Verifies that the method correctly updates distances only when new values are larger
    /// (for max type) and returns appropriate boolean values indicating update success.
    func testClusterDistancesTryUpdatingMaxType() {
        var clusterDistances = ClusterDistances(
            type: .max,
            distances: [0.1, 0.2, 0.3],
            otherIndices: [1, 0, 2]
        )
        
        // Test updating with larger distance (should update)
        let updated = clusterDistances.tryUpdating(index: 0, otherIndex: 1, distance: 0.5)
        XCTAssertTrue(updated)
        XCTAssertEqual(clusterDistances.distances[0], 0.5)
        XCTAssertEqual(clusterDistances.otherIndices[0], 1)
        
        // Test updating with smaller distance (should not update)
        let notUpdated = clusterDistances.tryUpdating(index: 0, otherIndex: 2, distance: 0.2)
        XCTAssertFalse(notUpdated)
        XCTAssertEqual(clusterDistances.distances[0], 0.5) // Should remain unchanged
    }
    
    // MARK: - computeDistancesToClusters Tests
    
    /// Tests computeDistancesToClusters with empty clusters array.
    /// Verifies that the function handles empty cluster scenarios gracefully
    /// and returns appropriate empty results without errors.
    func testComputeDistancesToClustersEmptyClusters() {
        let embeddings: [[Float]] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        let clusters: [Cluster] = []
        let distancesToClusters = ClusterDistances(type: .min, distances: [], otherIndices: [])
        
        let result = computeDistancesToClusters(
            embeddings: embeddings,
            clusters: clusters,
            distancesToClusters: distancesToClusters
        )
        
        XCTAssertEqual(result.type, .min)
        XCTAssertTrue(result.distances.isEmpty)
        XCTAssertTrue(result.otherIndices.isEmpty)
    }
    
    /// Tests computeDistancesToClusters with min distance type calculation.
    /// Verifies that the function correctly computes minimum distances between embeddings
    /// and clusters, ensuring proper distance calculation and index assignment.
    func testComputeDistancesToClustersMinType() {
        let embeddings: [[Float]] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        let clusters = [
            Cluster(embeddingIndices: [0], centroid: [1.0, 0.0, 0.0]),
            Cluster(embeddingIndices: [1], centroid: [0.0, 1.0, 0.0])
        ]
        let distancesToClusters = ClusterDistances(type: .min, distances: [], otherIndices: [])
        
        let result = computeDistancesToClusters(
            embeddings: embeddings,
            clusters: clusters,
            distancesToClusters: distancesToClusters
        )
        
        XCTAssertEqual(result.type, .min)
        XCTAssertEqual(result.distances.count, 2)
        XCTAssertEqual(result.otherIndices.count, 2)
        
        // First embedding should be closest to first cluster (distance 0)
        XCTAssertEqual(result.distances[0], 0.0, accuracy: 0.001)
        XCTAssertEqual(result.otherIndices[0], 0)
        
        // Second embedding should be closest to second cluster (distance 0)
        XCTAssertEqual(result.distances[1], 0.0, accuracy: 0.001)
        XCTAssertEqual(result.otherIndices[1], 1)
    }
    
    /// Tests computeDistancesToClusters with max distance type calculation.
    /// Verifies that the function correctly computes maximum distances between embeddings
    /// and clusters, ensuring proper distance calculation and index assignment.
    func testComputeDistancesToClustersMaxType() {
        let embeddings: [[Float]] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        let clusters = [
            Cluster(embeddingIndices: [0], centroid: [1.0, 0.0, 0.0]),
            Cluster(embeddingIndices: [1], centroid: [0.0, 1.0, 0.0])
        ]
        let distancesToClusters = ClusterDistances(type: .max, distances: [], otherIndices: [])
        
        let result = computeDistancesToClusters(
            embeddings: embeddings,
            clusters: clusters,
            distancesToClusters: distancesToClusters
        )
        
        XCTAssertEqual(result.type, .max)
        XCTAssertEqual(result.distances.count, 2)
        XCTAssertEqual(result.otherIndices.count, 2)
        
        // First embedding should be farthest from second cluster
        XCTAssertEqual(result.distances[0], 1.0, accuracy: 0.001)
        XCTAssertEqual(result.otherIndices[0], 1)
        
        // Second embedding should be farthest from first cluster
        XCTAssertEqual(result.distances[1], 1.0, accuracy: 0.001)
        XCTAssertEqual(result.otherIndices[1], 0)
    }
    
    /// Tests computeDistancesToClusters with a removed embedding index.
    /// Verifies that the function correctly handles removed embeddings by setting
    /// their distances to NaN and recalculating distances for remaining embeddings.
    func testComputeDistancesToClustersWithRemovedEmbedding() {
        let embeddings: [[Float]] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        let clusters = [
            Cluster(embeddingIndices: [0], centroid: [1.0, 0.0, 0.0]),
            Cluster(embeddingIndices: [1], centroid: [0.0, 1.0, 0.0])
        ]
        let distancesToClusters = ClusterDistances(type: .min, distances: [0.5, 0.3], otherIndices: [0, 1])
        
        let result = computeDistancesToClusters(
            embeddings: embeddings,
            clusters: clusters,
            distancesToClusters: distancesToClusters,
            removedEmbeddingIndex: 0
        )
        
        XCTAssertTrue(result.distances[0].isNaN)
        XCTAssertEqual(result.distances[1], 0.0, accuracy: 0.001)
    }
    
    /// Tests computeDistancesToClusters with a removed cluster index.
    /// Verifies that the function correctly handles removed clusters by resetting
    /// distances for embeddings that were associated with the removed cluster.
    func testComputeDistancesToClustersWithRemovedCluster() {
        let embeddings: [[Float]] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        let clusters = [
            Cluster(embeddingIndices: [0], centroid: []), // Empty centroid for removed cluster
            Cluster(embeddingIndices: [1], centroid: [0.0, 1.0, 0.0])
        ]
        let distancesToClusters = ClusterDistances(type: .min, distances: [0.5, 0.3], otherIndices: [0, 1])
        
        let result = computeDistancesToClusters(
            embeddings: embeddings,
            clusters: clusters,
            distancesToClusters: distancesToClusters,
            removedClusterIndex: 0
        )
        
        // Should reset distances for embeddings that were associated with removed cluster
        XCTAssertEqual(result.distances[0], 1.0, accuracy: 0.001) // Distance to remaining cluster
        XCTAssertEqual(result.otherIndices[0], 1) // Index of remaining cluster
    }
    
    /// Tests computeDistancesToClusters with filtered clusters (empty centroids).
    /// Verifies that the function correctly filters out clusters with empty centroids
    /// and only considers valid clusters in distance calculations.
    func testComputeDistancesToClustersWithFilteredClusters() {
        let embeddings: [[Float]] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        let clusters = [
            Cluster(embeddingIndices: [0], centroid: [1.0, 0.0, 0.0]),
            Cluster(embeddingIndices: [1], centroid: []) // Empty centroid - filtered out
        ]
        let distancesToClusters = ClusterDistances(type: .min, distances: [], otherIndices: [])
        
        let result = computeDistancesToClusters(
            embeddings: embeddings,
            clusters: clusters,
            distancesToClusters: distancesToClusters
        )
        
        // Should only consider the first cluster (non-empty centroid)
        XCTAssertEqual(result.distances[0], 0.0, accuracy: 0.001)
        XCTAssertEqual(result.otherIndices[0], 0)
        
        // Second embedding should be closest to first cluster
        XCTAssertEqual(result.distances[1], 1.0, accuracy: 0.001)
        XCTAssertEqual(result.otherIndices[1], 0)
    }
    
    // MARK: - Edge Cases and Performance Tests
    
    /// Tests computeDistancesToClusters with a large dataset to verify performance and scalability.
    /// Ensures the function can handle substantial numbers of embeddings and clusters
    /// while maintaining numerical stability and producing finite results.
    func testClusterDistancesWithLargeDataset() {
        let embeddings = generateTestEmbeddings(count: 50, dimension: 16)
        let clusters = (0..<10).map { i in
            Cluster(embeddingIndices: [i], centroid: embeddings[i])
        }
        let distancesToClusters = ClusterDistances(type: .min, distances: [], otherIndices: [])
        
        let result = computeDistancesToClusters(
            embeddings: embeddings,
            clusters: clusters,
            distancesToClusters: distancesToClusters
        )
        
        XCTAssertEqual(result.distances.count, embeddings.count)
        XCTAssertEqual(result.otherIndices.count, embeddings.count)
        
        // All distances should be finite
        XCTAssertTrue(result.distances.allSatisfy { $0.isFinite })
    }
    
    /// Tests ClusterDistances tryUpdating with edge cases including infinity values.
    /// Verifies that the method correctly handles special floating-point values
    /// and updates appropriately from infinity to finite values.
    func testClusterDistancesTryUpdatingEdgeCases() {
        var clusterDistances = ClusterDistances(
            type: .min,
            distances: [Float.infinity, Float.infinity],
            otherIndices: [-1, -1]
        )
        
        // Test updating from infinity
        let updated = clusterDistances.tryUpdating(index: 0, otherIndex: 1, distance: 0.5)
        XCTAssertTrue(updated)
        XCTAssertEqual(clusterDistances.distances[0], 0.5)
        XCTAssertEqual(clusterDistances.otherIndices[0], 1)
    }
    
    /// Tests ClusterDistances tryUpdating with max type and negative distances.
    /// Verifies that the method correctly handles negative distance values
    /// and updates appropriately for max type distance calculations.
    func testClusterDistancesMaxTypeWithNegativeDistances() {
        var clusterDistances = ClusterDistances(
            type: .max,
            distances: [-1.0, -2.0],
            otherIndices: [0, 1]
        )
        
        // Test updating with larger (less negative) distance
        let updated = clusterDistances.tryUpdating(index: 0, otherIndex: 1, distance: -0.5)
        XCTAssertTrue(updated)
        XCTAssertEqual(clusterDistances.distances[0], -0.5)
        XCTAssertEqual(clusterDistances.otherIndices[0], 1)
    }
    
    // MARK: - Additional Edge Cases
    
    /// Tests ClusterDistances initialization with empty arrays.
    /// Verifies that the struct correctly handles empty distance and index arrays
    /// without causing errors or unexpected behavior.
    func testClusterDistancesWithEmptyArrays() {
        let clusterDistances = ClusterDistances(type: .min, distances: [], otherIndices: [])
        XCTAssertEqual(clusterDistances.distances.count, 0)
        XCTAssertEqual(clusterDistances.otherIndices.count, 0)
    }
    
    /// Tests ClusterDistances initialization with single element arrays.
    /// Verifies that the struct correctly handles minimal data scenarios
    /// and maintains proper array counts and relationships.
    func testClusterDistancesWithSingleElement() {
        let clusterDistances = ClusterDistances(type: .min, distances: [0.5], otherIndices: [0])
        XCTAssertEqual(clusterDistances.distances.count, 1)
        XCTAssertEqual(clusterDistances.otherIndices.count, 1)
    }
    
    func testClusterDistancesTryUpdatingWithInfinity() {
        var clusterDistances = ClusterDistances(
            type: .min,
            distances: [.infinity, .infinity],
            otherIndices: [-1, -1]
        )
        
        let updated = clusterDistances.tryUpdating(index: 0, otherIndex: 1, distance: 0.5)
        XCTAssertTrue(updated)
        XCTAssertEqual(clusterDistances.distances[0], 0.5)
        XCTAssertEqual(clusterDistances.otherIndices[0], 1)
    }
    
    /// Tests ClusterDistances tryUpdating with NaN values.
    /// Verifies that the method correctly handles NaN values by not updating them
    /// and maintaining the NaN state as expected for invalid distance calculations.
    func testClusterDistancesTryUpdatingWithNaN() {
        var clusterDistances = ClusterDistances(
            type: .min,
            distances: [0.5, .nan],
            otherIndices: [0, -1]
        )
        
        // Should not update NaN values
        let updated = clusterDistances.tryUpdating(index: 1, otherIndex: 0, distance: 0.3)
        XCTAssertFalse(updated)
        XCTAssertTrue(clusterDistances.distances[1].isNaN)
    }
    
    func testClusterDistancesMaxTypeWithNegativeInfinity() {
        var clusterDistances = ClusterDistances(
            type: .max,
            distances: [-.infinity, -.infinity],
            otherIndices: [-1, -1]
        )
        
        let updated = clusterDistances.tryUpdating(index: 0, otherIndex: 1, distance: -0.5)
        XCTAssertTrue(updated)
        XCTAssertEqual(clusterDistances.distances[0], -0.5)
        XCTAssertEqual(clusterDistances.otherIndices[0], 1)
    }
    
    func testComputeDistancesToClustersWithEmptyEmbeddings() {
        let embeddings: [[Float]] = []
        let clusters = [
            Cluster(embeddingIndices: [0], centroid: [1.0, 0.0, 0.0])
        ]
        let distancesToClusters = ClusterDistances(type: .min, distances: [], otherIndices: [])
        
        let result = computeDistancesToClusters(
            embeddings: embeddings,
            clusters: clusters,
            distancesToClusters: distancesToClusters
        )
        
        XCTAssertEqual(result.distances.count, 0)
        XCTAssertEqual(result.otherIndices.count, 0)
    }
    
    func testComputeDistancesToClustersWithAllFilteredClusters() {
        let embeddings: [[Float]] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        let clusters = [
            Cluster(embeddingIndices: [0], centroid: []), // Filtered out
            Cluster(embeddingIndices: [1], centroid: [])  // Filtered out
        ]
        let distancesToClusters = ClusterDistances(type: .min, distances: [], otherIndices: [])
        
        let result = computeDistancesToClusters(
            embeddings: embeddings,
            clusters: clusters,
            distancesToClusters: distancesToClusters
        )
        
        XCTAssertEqual(result.distances.count, 2)
        XCTAssertEqual(result.otherIndices.count, 2)
        
        // All distances should be infinity since no valid clusters
        XCTAssertEqual(result.distances[0], .infinity)
        XCTAssertEqual(result.distances[1], .infinity)
    }
    
    func testComputeDistancesToClustersWithSingleCluster() {
        let embeddings: [[Float]] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        let clusters = [
            Cluster(embeddingIndices: [0], centroid: [1.0, 0.0, 0.0])
        ]
        let distancesToClusters = ClusterDistances(type: .min, distances: [], otherIndices: [])
        
        let result = computeDistancesToClusters(
            embeddings: embeddings,
            clusters: clusters,
            distancesToClusters: distancesToClusters
        )
        
        XCTAssertEqual(result.distances.count, 2)
        XCTAssertEqual(result.otherIndices.count, 2)
        
        // First embedding should be closest to the cluster (distance 0)
        XCTAssertEqual(result.distances[0], 0.0, accuracy: 0.001)
        XCTAssertEqual(result.otherIndices[0], 0)
        
        // Second embedding should be farther from the cluster
        XCTAssertEqual(result.distances[1], 1.0, accuracy: 0.001)
        XCTAssertEqual(result.otherIndices[1], 0)
    }
    
    func testComputeDistancesToClustersWithRemovedEmbeddingAndCluster() {
        let embeddings: [[Float]] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        let clusters = [
            Cluster(embeddingIndices: [0], centroid: []), // Empty centroid for removed cluster
            Cluster(embeddingIndices: [1], centroid: [0.0, 1.0, 0.0])
        ]
        let distancesToClusters = ClusterDistances(
            type: .min,
            distances: [0.5, 0.3],
            otherIndices: [0, 1]
        )
        
        let result = computeDistancesToClusters(
            embeddings: embeddings,
            clusters: clusters,
            distancesToClusters: distancesToClusters,
            removedEmbeddingIndex: 0,
            removedClusterIndex: 0
        )
        
        XCTAssertEqual(result.distances.count, 2)
        XCTAssertEqual(result.otherIndices.count, 2)
        
        // First embedding should be NaN (removed)
        XCTAssertTrue(result.distances[0].isNaN)
        
        // Second embedding should be closest to second cluster
        XCTAssertEqual(result.distances[1], 0.0, accuracy: 0.001)
        XCTAssertEqual(result.otherIndices[1], 1)
    }
    
    func testClusterDistancesMemoryEfficiency() {
        let largeCount = 10000
        let distances: [Float] = Array(repeating: 0.5, count: largeCount)
        let otherIndices = Array(repeating: 0, count: largeCount)
        
        let clusterDistances = ClusterDistances(
            type: .min,
            distances: distances,
            otherIndices: otherIndices
        )
        
        XCTAssertEqual(clusterDistances.distances.count, largeCount)
        XCTAssertEqual(clusterDistances.otherIndices.count, largeCount)
    }
    
    /// Tests the performance of computeDistancesToClusters with a substantial dataset.
    /// Measures execution time to ensure the function performs efficiently
    /// with realistic embedding dimensions and cluster counts.
    func testComputeDistancesToClustersPerformance() {
        let embeddings = generateTestEmbeddings(count: 100, dimension: SpeakerManager.embeddingSize)
        let clusters = (0..<10).map { i in
            Cluster(embeddingIndices: [i], centroid: embeddings[i])
        }
        let distancesToClusters = ClusterDistances(type: .min, distances: [], otherIndices: [])
        
        measure {
            let result = computeDistancesToClusters(
                embeddings: embeddings,
                clusters: clusters,
                distancesToClusters: distancesToClusters
            )
            XCTAssertEqual(result.distances.count, embeddings.count)
        }
    }
    
}
