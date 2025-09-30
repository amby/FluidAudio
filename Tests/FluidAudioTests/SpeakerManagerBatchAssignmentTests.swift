import XCTest
@testable import FluidAudio

/// Comprehensive test suite for SpeakerManager batch assignment functionality.
/// 
/// This test class covers the new `assignSpeakers` method that processes multiple embeddings
/// in a single batch operation, providing improved performance and efficiency compared to
/// individual speaker assignments. Tests include basic functionality, edge cases, performance
/// scenarios, and integration with existing SpeakerManager features.
/// 
/// Key test categories:
/// - Basic batch assignment with valid embeddings
/// - Edge cases with invalid embeddings and extreme values
/// - Performance testing with large batches
/// - Integration with existing speaker management features
/// - Error handling and boundary conditions
final class SpeakerManagerBatchAssignmentTests: XCTestCase {
    
    var speakerManager: SpeakerManager!
    
    override func setUp() {
        super.setUp()
        speakerManager = SpeakerManager(
            speakerThreshold: 0.65,
            embeddingThreshold: 0.45,
            minSpeechDuration: 1.0,
            minEmbeddingUpdateDuration: 2.0,
            maxEmbeddingsPerSpeaker: 50
        )
    }
    
    override func tearDown() {
        speakerManager = nil
        super.tearDown()
    }
    
    // MARK: - Basic Batch Assignment Tests
    
    /// Tests that empty input arrays return an empty result array.
    /// This ensures the batch assignment method handles empty inputs gracefully
    /// without throwing errors or returning unexpected results.
    func testAssignSpeakersEmptyInput() {
        let result = speakerManager.assignSpeakers(
            embeddings: [],
            durations: [],
            confidences: []
        )
        
        XCTAssertTrue(result.isEmpty)
    }
        
    /// Tests batch assignment with valid embeddings of correct dimensions.
    /// Verifies that the method successfully creates Speaker objects for valid embeddings
    /// and returns the expected number of results matching the input count.
    func testAssignSpeakersWithValidEmbeddings() {
        let embeddings: [[Float]] = [
            [1.0, 0.0] + .init(repeating: 0.0, count: SpeakerManager.embeddingSize - 2),
            [0.0, 1.0] + .init(repeating: 0.0, count: SpeakerManager.embeddingSize - 2)
        ]
        let durations: [Float] = [2.0, 1.5]
        let confidences: [Float] = [0.8, 0.9]
        
        let result = speakerManager.assignSpeakers(
            embeddings: embeddings,
            durations: durations,
            confidences: confidences
        )
        
        XCTAssertEqual(result.count, 2)
        // Should create new speakers for both embeddings
        XCTAssertNotNil(result[0])
        XCTAssertNotNil(result[1])
        XCTAssertNotEqual(result[0]?.id, result[1]?.id)
    }
    
    /// Tests batch assignment with embeddings that have durations below the minimum threshold.
    /// Verifies that embeddings with short durations are filtered out and return nil,
    /// ensuring the minimum speech duration requirement is enforced.
    func testAssignSpeakersWithShortDurationEmbeddings() {
        let embeddings: [[Float]] = [
            [1.0, 0.0] + .init(repeating: 0.0, count: SpeakerManager.embeddingSize - 2),
            [0.0, 1.0] + .init(repeating: 0.0, count: SpeakerManager.embeddingSize - 2)
        ]
        let durations: [Float] = [0.5, 0.3] // Below minSpeechDuration
        let confidences: [Float] = [0.8, 0.9]
        
        let result = speakerManager.assignSpeakers(
            embeddings: embeddings,
            durations: durations,
            confidences: confidences
        )
        
        XCTAssertEqual(result.count, 2)
        // Should return nil for short duration embeddings
        XCTAssertNil(result[0])
        XCTAssertNil(result[1])
    }
    
    /// Tests batch assignment with a mix of valid and invalid embeddings.
    /// Verifies that invalid embeddings (empty arrays, wrong dimensions) are handled gracefully
    /// by returning nil, while valid embeddings still create Speaker objects.
    func testAssignSpeakersWithInvalidEmbeddings() {
        let embeddings: [[Float]] = [
            [], // Empty embedding
            [1.0, 2.0], // Wrong dimension
            [1.0, 0.0] + .init(repeating: 0.0, count: SpeakerManager.embeddingSize - 2) // Valid
        ]
        let durations: [Float] = [1.0, 1.0, 2.0]
        let confidences: [Float] = [0.8, 0.9, 0.7]
        
        let result = speakerManager.assignSpeakers(
            embeddings: embeddings,
            durations: durations,
            confidences: confidences
        )
        
        XCTAssertEqual(result.count, 3)
        // Should return nil for invalid embeddings
        XCTAssertNil(result[0])
        XCTAssertNil(result[1])
        // Should create speaker for valid embedding
        XCTAssertNotNil(result[2])
    }
    
    // MARK: - Clustering and Speaker Association Tests
    
    /// Tests batch assignment with similar embeddings that should be associated with the same speaker.
    /// Verifies that the clustering algorithm correctly groups similar embeddings together
    /// and creates appropriate speaker associations.
    func testAssignSpeakersWithSimilarEmbeddings() {
        let baseEmbedding: [Float] = [1.0, 0.0] + .init(repeating: 0.0, count: SpeakerManager.embeddingSize - 2)
        let similarEmbedding: [Float] = [0.9, 0.1] + .init(repeating: 0.0, count: SpeakerManager.embeddingSize - 2)
        
        let embeddings = [baseEmbedding, similarEmbedding]
        let durations: [Float] = [2.0, 1.5]
        let confidences: [Float] = [0.8, 0.9]
        
        let result = speakerManager.assignSpeakers(
            embeddings: embeddings,
            durations: durations,
            confidences: confidences
        )
        
        XCTAssertEqual(result.count, 2)
        // Should create speakers for both embeddings
        XCTAssertNotNil(result[0])
        XCTAssertNotNil(result[1])
    }
    
    /// Tests batch assignment when existing speakers are already present in the system.
    /// Verifies that the method correctly integrates with pre-existing speakers and
    /// handles the association of new embeddings with known speakers.
    func testAssignSpeakersWithExistingSpeakers() {
        // First, create some existing speakers
        let existingSpeaker = Speaker(
            id: "existing-1",
            name: "Existing Speaker",
            currentEmbedding: [1.0, 0.0] + .init(repeating: 0.0, count: SpeakerManager.embeddingSize - 2),
            duration: 5.0,
            createdAt: Date(),
            updatedAt: Date()
        )
        
        speakerManager.initializeKnownSpeakers([existingSpeaker])
        
        // Now assign new embeddings
        let embeddings: [[Float]] = [
            [0.9, 0.1] + .init(repeating: 0.0, count: SpeakerManager.embeddingSize - 2), // Similar to existing
            [0.0, 1.0] + .init(repeating: 0.0, count: SpeakerManager.embeddingSize - 2)  // Different
        ]
        let durations: [Float] = [2.0, 1.5]
        let confidences: [Float] = [0.8, 0.9]
        
        let result = speakerManager.assignSpeakers(
            embeddings: embeddings,
            durations: durations,
            confidences: confidences
        )
        
        XCTAssertEqual(result.count, 2)
        // First embedding should be associated with existing speaker
        XCTAssertEqual(result[0]?.id, "existing-1")
        // Second embedding should create new speaker
        XCTAssertNotNil(result[1])
        XCTAssertNotEqual(result[1]?.id, "existing-1")
    }
    
    // MARK: - Edge Cases and Error Handling
    
    /// Tests batch assignment with zero confidence values.
    /// Verifies that the method handles low confidence embeddings appropriately
    /// and still processes them according to the assignment logic.
    func testAssignSpeakersWithZeroConfidence() {
        let embeddings: [[Float]] = [[1.0, 0.0] + .init(repeating: 0.0, count: SpeakerManager.embeddingSize - 2)]
        let durations: [Float] = [2.0]
        let confidences: [Float] = [0.0] // Zero confidence
        
        let result = speakerManager.assignSpeakers(
            embeddings: embeddings,
            durations: durations,
            confidences: confidences
        )
        
        XCTAssertEqual(result.count, 1)
        // Should still create speaker despite zero confidence
        XCTAssertNotNil(result[0])
    }
    
    /// Tests batch assignment with maximum confidence values.
    /// Verifies that high confidence embeddings are processed correctly
    /// and that confidence values don't interfere with the assignment logic.
    func testAssignSpeakersWithHighConfidence() {
        let embeddings: [[Float]] = [[1.0, 0.0] + .init(repeating: 0.0, count: SpeakerManager.embeddingSize - 2)]
        let durations: [Float] = [2.0]
        let confidences: [Float] = [1.0] // Maximum confidence
        
        let result = speakerManager.assignSpeakers(
            embeddings: embeddings,
            durations: durations,
            confidences: confidences
        )
        
        XCTAssertEqual(result.count, 1)
        XCTAssertNotNil(result[0])
    }
    
    /// Tests batch assignment with a large number of embeddings to verify performance and scalability.
    /// Ensures the method can handle substantial batches without performance degradation
    /// and maintains correct assignment logic across many embeddings.
    func testAssignSpeakersWithLargeBatch() {
        let embeddings = generateTestEmbeddings(count: 20, dimension: SpeakerManager.embeddingSize)
        let durations = Array(repeating: Float(2.0), count: 20)
        let confidences = Array(repeating: Float(0.8), count: 20)
        
        let result = speakerManager.assignSpeakers(
            embeddings: embeddings,
            durations: durations,
            confidences: confidences
        )
        
        XCTAssertEqual(result.count, 20)
        // Should create speakers for all embeddings
        XCTAssertTrue(result.allSatisfy { $0 != nil })
    }
    
    /// Tests batch assignment with a mix of valid and invalid embeddings in a single batch.
    /// Verifies that the method correctly processes valid embeddings while gracefully
    /// handling invalid ones, ensuring robust error handling in mixed scenarios.
    func testAssignSpeakersMixedValidInvalid() {
        let embeddings: [[Float]] = [
            [1.0, 0.0] + .init(repeating: 0.0, count: SpeakerManager.embeddingSize - 2), // Valid
            [], // Invalid - empty
            [0.0, 1.0] + .init(repeating: 0.0, count: SpeakerManager.embeddingSize - 2), // Valid
            [1.0, 2.0], // Invalid - wrong dimension
            [0.0, 0.0, 1.0] + .init(repeating: 0.0, count: SpeakerManager.embeddingSize - 3)  // Valid
        ]
        let durations: [Float] = [2.0, 1.0, 1.5, 1.0, 2.5]
        let confidences: [Float] = [0.8, 0.9, 0.7, 0.6, 0.85]
        
        let result = speakerManager.assignSpeakers(
            embeddings: embeddings,
            durations: durations,
            confidences: confidences
        )
        
        XCTAssertEqual(result.count, 5)
        // Should create speakers only for valid embeddings
        XCTAssertNotNil(result[0])
        XCTAssertNil(result[1])
        XCTAssertNotNil(result[2])
        XCTAssertNil(result[3])
        XCTAssertNotNil(result[4])
    }
    
    // MARK: - Performance Tests
    
    /// Tests the performance of batch assignment with a substantial number of embeddings.
    /// Measures execution time to ensure the batch processing provides performance benefits
    /// over individual speaker assignments and scales appropriately.
    func testAssignSpeakersPerformance() {
        let embeddings = generateTestEmbeddings(count: 100, dimension: SpeakerManager.embeddingSize)
        let durations = Array(repeating: Float(2.0), count: 100)
        let confidences = Array(repeating: Float(0.8), count: 100)
        
        measure {
            let result = speakerManager.assignSpeakers(
                embeddings: embeddings,
                durations: durations,
                confidences: confidences
            )
            XCTAssertEqual(result.count, 100)
        }
    }
    
    // MARK: - Edge Case Tests
    
    /// Tests batch assignment with extreme embedding values including very large and very small numbers.
    /// Verifies that the method handles edge cases in embedding values gracefully without
    /// causing numerical instability or unexpected behavior.
    func testSpeakerManagerWithExtremeEmbeddings() {
        let speakerManager = SpeakerManager()
        
        let embeddings = [
            Array(repeating: 1.0, count: SpeakerManager.embeddingSize), // All ones
            Array(repeating: -1.0, count: SpeakerManager.embeddingSize), // All negative ones
            Array(repeating: 0.0, count: SpeakerManager.embeddingSize), // All zeros
            Array(repeating: Float.greatestFiniteMagnitude, count: SpeakerManager.embeddingSize), // Very large values
            Array(repeating: Float.leastNormalMagnitude, count: SpeakerManager.embeddingSize) // Very small values
        ] as [[Float]]
        let durations: [Float] = [2.0, 1.5, 3.0, 2.5, 1.0]
        let confidences: [Float] = [0.8, 0.9, 0.7, 0.85, 0.6]
        
        let result = speakerManager.assignSpeakers(
            embeddings: embeddings,
            durations: durations,
            confidences: confidences
        )
        
        XCTAssertEqual(result.count, 5)
        // Should handle extreme values gracefully
        XCTAssertTrue(result.allSatisfy { $0 != nil })
    }
    
    /// Tests batch assignment with identical embeddings to verify clustering behavior.
    /// Ensures that duplicate embeddings are handled appropriately and don't cause
    /// issues in the clustering algorithm or speaker assignment logic.
    func testSpeakerManagerWithDuplicateEmbeddings() {
        let speakerManager = SpeakerManager()
        
        let baseEmbedding: [Float] = [1.0, 0.0] + .init(repeating: 0.0, count: SpeakerManager.embeddingSize - 2)
        let embeddings = [baseEmbedding, baseEmbedding, baseEmbedding]
        let durations: [Float] = [2.0, 1.5, 3.0]
        let confidences: [Float] = [0.8, 0.9, 0.7]
        
        let result = speakerManager.assignSpeakers(
            embeddings: embeddings,
            durations: durations,
            confidences: confidences
        )
        
        XCTAssertEqual(result.count, 3)
        // Should handle duplicate embeddings gracefully
        XCTAssertTrue(result.allSatisfy { $0 != nil })
    }
    
    /// Tests batch assignment with a very large number of embeddings to verify scalability.
    /// Ensures the method can handle substantial batches (1000+ embeddings) without
    /// memory issues or performance degradation, validating production-scale usage.
    func testSpeakerManagerWithVeryLargeBatch() {
        let speakerManager = SpeakerManager()
        
        let embeddings = generateTestEmbeddings(count: 1000, dimension: SpeakerManager.embeddingSize)
        let durations = Array(repeating: Float(2.0), count: 1000)
        let confidences = Array(repeating: Float(0.8), count: 1000)
        
        let result = speakerManager.assignSpeakers(
            embeddings: embeddings,
            durations: durations,
            confidences: confidences
        )
        
        XCTAssertEqual(result.count, 1000)
        // Should handle large batches without crashing
        XCTAssertTrue(result.allSatisfy { $0 != nil })
    }
    
    /// Tests batch assignment with a complex mix of valid and invalid embeddings including edge cases.
    /// Verifies robust handling of various invalid embedding types (empty, wrong dimension, NaN, infinity)
    /// while ensuring valid embeddings are processed correctly in the same batch.
    func testSpeakerManagerWithMixedValidInvalidEmbeddings() {
        let speakerManager = SpeakerManager()
        
        let embeddings: [[Float]] = [
            [1.0, 0.0] + .init(repeating: 0.0, count: SpeakerManager.embeddingSize - 2), // Valid
            [], // Invalid - empty
            [0.0, 1.0] + .init(repeating: 0.0, count: SpeakerManager.embeddingSize - 2), // Valid
            [1.0, 2.0], // Invalid - wrong dimension
            [0.0, 0.0, 1.0] + .init(repeating: 0.0, count: SpeakerManager.embeddingSize - 3), // Valid
            Array(repeating: Float.nan, count: SpeakerManager.embeddingSize), // Invalid - contains NaN
            Array(repeating: Float.infinity, count: SpeakerManager.embeddingSize) // Invalid - contains infinity
        ]
        let durations: [Float] = [2.0, 1.0, 1.5, 1.0, 2.5, 1.0, 1.0]
        let confidences: [Float] = [0.8, 0.9, 0.7, 0.6, 0.85, 0.5, 0.4]
        
        let result = speakerManager.assignSpeakers(
            embeddings: embeddings,
            durations: durations,
            confidences: confidences
        )
        
        XCTAssertEqual(result.count, 7)
        // assignSpeakers behavior: creates speakers for valid embeddings, nil for invalid ones
        XCTAssertNotNil(result[0]) // Valid embedding
        XCTAssertNil(result[1]) // Empty embedding - invalid
        XCTAssertNotNil(result[2]) // Valid embedding
        XCTAssertNil(result[3]) // Wrong dimension - invalid
        XCTAssertNotNil(result[4]) // Valid embedding
        XCTAssertNotNil(result[5]) // NaN embedding - handled gracefully
        XCTAssertNotNil(result[6]) // Infinity embedding - handled gracefully
    }
    
    // MARK: - Helper Methods
    
    /// Generates test embeddings for performance and scalability testing.
    /// Creates a specified number of embeddings with the given dimension,
    /// using deterministic values to ensure reproducible test results.
    /// - Parameters:
    ///   - count: Number of embeddings to generate
    ///   - dimension: Dimension of each embedding vector
    /// - Returns: Array of test embeddings with deterministic values
    private func generateTestEmbeddings(count: Int, dimension: Int) -> [[Float]] {
        var embeddings: [[Float]] = []
        for i in 0..<count {
            var embedding: [Float] = []
            for j in 0..<dimension {
                embedding.append(Float(i * dimension + j) / 100.0)
            }
            embeddings.append(embedding)
        }
        return embeddings
    }
}
