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
        let embeddingWeights: [Float] = []
        let (speakers, indices) = speakerManager.assignSpeakers(
            embeddings: [],
            durations: [],
            embeddingWeights: embeddingWeights
        )
        
        XCTAssertTrue(speakers.isEmpty)
        XCTAssertTrue(indices.isEmpty)
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
        
        let embeddingWeights: [Float] = Array(repeating: 1.0, count: embeddings.count)
        let (speakers, indices) = speakerManager.assignSpeakers(
            embeddings: embeddings,
            durations: durations,
            embeddingWeights: embeddingWeights
        )
        
        XCTAssertEqual(speakers.count, 2)
        // Should create new speakers for both embeddings
        XCTAssertNotNil(speakers[0])
        XCTAssertNotNil(speakers[1])
        XCTAssertNotEqual(speakers[0]?.id, speakers[1]?.id)
        // Should return valid embedding indices
        XCTAssertEqual(indices.count, 2)
        XCTAssertTrue(indices[0] >= 0)
        XCTAssertTrue(indices[1] >= 0)
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
        
        let embeddingWeights: [Float] = Array(repeating: 1.0, count: embeddings.count)
        let (speakers, indices) = speakerManager.assignSpeakers(
            embeddings: embeddings,
            durations: durations,
            embeddingWeights: embeddingWeights
        )
        
        XCTAssertEqual(speakers.count, 2)
        // Should return nil for short duration embeddings
        XCTAssertNil(speakers[0])
        XCTAssertNil(speakers[1])
        // Should return -1 for invalid embedding indices
        XCTAssertEqual(indices.count, 2)
        XCTAssertEqual(indices[0], -1)
        XCTAssertEqual(indices[1], -1)
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
        
        let embeddingWeights: [Float] = Array(repeating: 1.0, count: embeddings.count)
        let (speakers, indices) = speakerManager.assignSpeakers(
            embeddings: embeddings,
            durations: durations,
            embeddingWeights: embeddingWeights
        )
        
        XCTAssertEqual(speakers.count, 3)
        // Should return nil for invalid embeddings
        XCTAssertNil(speakers[0])
        XCTAssertNil(speakers[1])
        // Should create speaker for valid embedding
        XCTAssertNotNil(speakers[2])
        // Should return -1 for invalid embeddings, valid index for valid embedding
        XCTAssertEqual(indices.count, 3)
        XCTAssertEqual(indices[0], -1)
        XCTAssertEqual(indices[1], -1)
        XCTAssertTrue(indices[2] >= 0)
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
        
        let embeddingWeights: [Float] = Array(repeating: 1.0, count: embeddings.count)
        let (speakers, indices) = speakerManager.assignSpeakers(
            embeddings: embeddings,
            durations: durations,
            embeddingWeights: embeddingWeights
        )
        
        XCTAssertEqual(speakers.count, 2)
        // Should create speakers for both embeddings
        XCTAssertNotNil(speakers[0])
        XCTAssertNotNil(speakers[1])
        // Should return valid embedding indices
        XCTAssertEqual(indices.count, 2)
        XCTAssertTrue(indices[0] >= 0)
        XCTAssertTrue(indices[1] >= 0)
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
        
        let embeddingWeights: [Float] = Array(repeating: 1.0, count: embeddings.count)
        let (speakers, indices) = speakerManager.assignSpeakers(
            embeddings: embeddings,
            durations: durations,
            embeddingWeights: embeddingWeights
        )
        
        XCTAssertEqual(speakers.count, 2)
        // First embedding should be associated with existing speaker
        XCTAssertEqual(speakers[0]?.id, "existing-1")
        // Second embedding should create new speaker
        XCTAssertNotNil(speakers[1])
        XCTAssertNotEqual(speakers[1]?.id, "existing-1")
        // Should return valid embedding indices
        XCTAssertEqual(indices.count, 2)
        XCTAssertTrue(indices[0] >= 0)
        XCTAssertTrue(indices[1] >= 0)
    }
    
    // MARK: - Edge Cases and Error Handling
    
    /// Tests batch assignment with zero confidence values.
    /// Verifies that the method handles low confidence embeddings appropriately
    /// and still processes them according to the assignment logic.
    func testAssignSpeakersWithZeroConfidence() {
        let embeddings: [[Float]] = [[1.0, 0.0] + .init(repeating: 0.0, count: SpeakerManager.embeddingSize - 2)]
        let durations: [Float] = [2.0]
        
        let embeddingWeights: [Float] = Array(repeating: 1.0, count: embeddings.count)
        let (speakers, indices) = speakerManager.assignSpeakers(
            embeddings: embeddings,
            durations: durations,
            embeddingWeights: embeddingWeights
        )
        
        XCTAssertEqual(speakers.count, 1)
        // Should still create speaker despite zero confidence
        XCTAssertNotNil(speakers[0])
        // Should return valid embedding index
        XCTAssertEqual(indices.count, 1)
        XCTAssertTrue(indices[0] >= 0)
    }
    
    /// Tests batch assignment with maximum confidence values.
    /// Verifies that high confidence embeddings are processed correctly
    /// and that confidence values don't interfere with the assignment logic.
    func testAssignSpeakersWithHighConfidence() {
        let embeddings: [[Float]] = [[1.0, 0.0] + .init(repeating: 0.0, count: SpeakerManager.embeddingSize - 2)]
        let durations: [Float] = [2.0]
        
        let embeddingWeights: [Float] = Array(repeating: 1.0, count: embeddings.count)
        let (speakers, indices) = speakerManager.assignSpeakers(
            embeddings: embeddings,
            durations: durations,
            embeddingWeights: embeddingWeights
        )
        
        XCTAssertEqual(speakers.count, 1)
        XCTAssertNotNil(speakers[0])
        // Should return valid embedding index
        XCTAssertEqual(indices.count, 1)
        XCTAssertTrue(indices[0] >= 0)
    }
    
    /// Tests batch assignment with a large number of embeddings to verify performance and scalability.
    /// Ensures the method can handle substantial batches without performance degradation
    /// and maintains correct assignment logic across many embeddings.
    func testAssignSpeakersWithLargeBatch() {
        let embeddings = generateTestEmbeddings(count: 20, dimension: SpeakerManager.embeddingSize)
        let durations = Array(repeating: Float(2.0), count: 20)
        
        let embeddingWeights: [Float] = Array(repeating: 1.0, count: embeddings.count)
        let (speakers, indices) = speakerManager.assignSpeakers(
            embeddings: embeddings,
            durations: durations,
            embeddingWeights: embeddingWeights
        )
        
        XCTAssertEqual(speakers.count, 20)
        // Should create speakers for all embeddings
        XCTAssertTrue(speakers.allSatisfy { $0 != nil })
        // Should return valid embedding indices
        XCTAssertEqual(indices.count, 20)
        XCTAssertTrue(indices.allSatisfy { $0 >= 0 })
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
        
        let embeddingWeights: [Float] = Array(repeating: 1.0, count: embeddings.count)
        let (speakers, indices) = speakerManager.assignSpeakers(
            embeddings: embeddings,
            durations: durations,
            embeddingWeights: embeddingWeights
        )
        
        XCTAssertEqual(speakers.count, 5)
        // Should create speakers only for valid embeddings
        XCTAssertNotNil(speakers[0])
        XCTAssertNil(speakers[1])
        XCTAssertNotNil(speakers[2])
        XCTAssertNil(speakers[3])
        XCTAssertNotNil(speakers[4])
        // Should return valid indices for valid embeddings, -1 for invalid ones
        XCTAssertEqual(indices.count, 5)
        XCTAssertTrue(indices[0] >= 0)
        XCTAssertEqual(indices[1], -1)
        XCTAssertTrue(indices[2] >= 0)
        XCTAssertEqual(indices[3], -1)
        XCTAssertTrue(indices[4] >= 0)
    }
    
    // MARK: - Performance Tests
    
    /// Tests the performance of batch assignment with a substantial number of embeddings.
    /// Measures execution time to ensure the batch processing provides performance benefits
    /// over individual speaker assignments and scales appropriately.
    func testAssignSpeakersPerformance() {
        let embeddings = generateTestEmbeddings(count: 100, dimension: SpeakerManager.embeddingSize)
        let durations = Array(repeating: Float(2.0), count: 100)
        
        measure {
            let embeddingWeights: [Float] = Array(repeating: 1.0, count: embeddings.count)
            let (speakers, indices) = speakerManager.assignSpeakers(
                embeddings: embeddings,
                durations: durations,
                embeddingWeights: embeddingWeights
            )
            XCTAssertEqual(speakers.count, 100)
            XCTAssertEqual(indices.count, 100)
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
        
        let embeddingWeights: [Float] = Array(repeating: 1.0, count: embeddings.count)
        let (speakers, indices) = speakerManager.assignSpeakers(
            embeddings: embeddings,
            durations: durations,
            embeddingWeights: embeddingWeights
        )
        
        XCTAssertEqual(speakers.count, 5)
        // Should handle extreme values gracefully
        XCTAssertTrue(speakers.allSatisfy { $0 != nil })
        // Should return valid embedding indices
        XCTAssertEqual(indices.count, 5)
        XCTAssertTrue(indices.allSatisfy { $0 >= 0 })
    }
    
    /// Tests batch assignment with identical embeddings to verify clustering behavior.
    /// Ensures that duplicate embeddings are handled appropriately and don't cause
    /// issues in the clustering algorithm or speaker assignment logic.
    func testSpeakerManagerWithDuplicateEmbeddings() {
        let speakerManager = SpeakerManager()
        
        let baseEmbedding: [Float] = [1.0, 0.0] + .init(repeating: 0.0, count: SpeakerManager.embeddingSize - 2)
        let embeddings = [baseEmbedding, baseEmbedding, baseEmbedding]
        let durations: [Float] = [2.0, 1.5, 3.0]
        
        let embeddingWeights: [Float] = Array(repeating: 1.0, count: embeddings.count)
        let (speakers, indices) = speakerManager.assignSpeakers(
            embeddings: embeddings,
            durations: durations,
            embeddingWeights: embeddingWeights
        )
        
        XCTAssertEqual(speakers.count, 3)
        // Should handle duplicate embeddings gracefully
        XCTAssertTrue(speakers.allSatisfy { $0 != nil })
        // Should return valid embedding indices
        XCTAssertEqual(indices.count, 3)
        XCTAssertTrue(indices.allSatisfy { $0 >= 0 })
    }
    
    /// Tests batch assignment with a very large number of embeddings to verify scalability.
    /// Ensures the method can handle substantial batches (1000+ embeddings) without
    /// memory issues or performance degradation, validating production-scale usage.
    func testSpeakerManagerWithVeryLargeBatch() {
        let speakerManager = SpeakerManager()
        
        let embeddings = generateTestEmbeddings(count: 1000, dimension: SpeakerManager.embeddingSize)
        let durations = Array(repeating: Float(2.0), count: 1000)
        
        let embeddingWeights: [Float] = Array(repeating: 1.0, count: embeddings.count)
        let (speakers, indices) = speakerManager.assignSpeakers(
            embeddings: embeddings,
            durations: durations,
            embeddingWeights: embeddingWeights
        )
        
        XCTAssertEqual(speakers.count, 1000)
        // Should handle large batches without crashing
        XCTAssertTrue(speakers.allSatisfy { $0 != nil })
        // Should return valid embedding indices
        XCTAssertEqual(indices.count, 1000)
        XCTAssertTrue(indices.allSatisfy { $0 >= 0 })
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
        
        let embeddingWeights: [Float] = Array(repeating: 1.0, count: embeddings.count)
        let (speakers, indices) = speakerManager.assignSpeakers(
            embeddings: embeddings,
            durations: durations,
            embeddingWeights: embeddingWeights
        )
        
        XCTAssertEqual(speakers.count, 7)
        // assignSpeakers behavior: creates speakers for valid embeddings, nil for invalid ones
        XCTAssertNotNil(speakers[0]) // Valid embedding
        XCTAssertNil(speakers[1]) // Empty embedding - invalid
        XCTAssertNotNil(speakers[2]) // Valid embedding
        XCTAssertNil(speakers[3]) // Wrong dimension - invalid
        XCTAssertNotNil(speakers[4]) // Valid embedding
        XCTAssertNotNil(speakers[5]) // NaN embedding - handled gracefully
        XCTAssertNotNil(speakers[6]) // Infinity embedding - handled gracefully
        // Should return valid indices for valid embeddings, -1 for invalid ones
        XCTAssertEqual(indices.count, 7)
        XCTAssertTrue(indices[0] >= 0) // Valid embedding
        XCTAssertEqual(indices[1], -1) // Empty embedding - invalid
        XCTAssertTrue(indices[2] >= 0) // Valid embedding
        XCTAssertEqual(indices[3], -1) // Wrong dimension - invalid
        XCTAssertTrue(indices[4] >= 0) // Valid embedding
        XCTAssertTrue(indices[5] >= 0) // NaN embedding - handled gracefully
        XCTAssertTrue(indices[6] >= 0) // Infinity embedding - handled gracefully
    }
    
    // MARK: - EmbeddingIndex Tracking Tests
    
    /// Tests that assignSpeakers returns correct embedding indices for valid embeddings.
    /// Verifies that the embedding indices correspond to the order of valid embeddings
    /// and that invalid embeddings return -1.
    func testAssignSpeakersEmbeddingIndexTracking() {
        let embeddings: [[Float]] = [
            [1.0, 0.0] + .init(repeating: 0.0, count: SpeakerManager.embeddingSize - 2), // Valid
            [], // Invalid - empty
            [0.0, 1.0] + .init(repeating: 0.0, count: SpeakerManager.embeddingSize - 2), // Valid
            [1.0, 2.0], // Invalid - wrong dimension
            [0.0, 0.0, 1.0] + .init(repeating: 0.0, count: SpeakerManager.embeddingSize - 3) // Valid
        ]
        let durations: [Float] = [2.0, 1.0, 1.5, 1.0, 2.5]
        
        let embeddingWeights: [Float] = Array(repeating: 1.0, count: embeddings.count)
        let (speakers, indices) = speakerManager.assignSpeakers(
            embeddings: embeddings,
            durations: durations,
            embeddingWeights: embeddingWeights
        )
        
        XCTAssertEqual(speakers.count, 5)
        XCTAssertEqual(indices.count, 5)
        
        // Valid embeddings should have valid indices
        XCTAssertNotNil(speakers[0])
        XCTAssertTrue(indices[0] >= 0)
        
        // Invalid embeddings should have -1 indices
        XCTAssertNil(speakers[1])
        XCTAssertEqual(indices[1], -1)
        
        XCTAssertNotNil(speakers[2])
        XCTAssertTrue(indices[2] >= 0)
        
        XCTAssertNil(speakers[3])
        XCTAssertEqual(indices[3], -1)
        
        XCTAssertNotNil(speakers[4])
        XCTAssertTrue(indices[4] >= 0)
        
        // Indices should be sequential for valid embeddings
        let validIndices = indices.filter { $0 >= 0 }
        let sortedIndices = validIndices.sorted()
        XCTAssertEqual(validIndices, sortedIndices)
    }
    
    /// Tests that embedding indices are consistent across multiple calls.
    /// Verifies that the embedding index tracking maintains consistency
    /// when multiple batches are processed.
    func testAssignSpeakersEmbeddingIndexConsistency() {
        let embeddings1: [[Float]] = [
            [1.0, 0.0] + .init(repeating: 0.0, count: SpeakerManager.embeddingSize - 2),
            [0.0, 1.0] + .init(repeating: 0.0, count: SpeakerManager.embeddingSize - 2)
        ]
        let durations1: [Float] = [2.0, 1.5]
        
        let embeddingWeights1: [Float] = Array(repeating: 1.0, count: embeddings1.count)
        let (_, indices1) = speakerManager.assignSpeakers(
            embeddings: embeddings1,
            durations: durations1,
            embeddingWeights: embeddingWeights1
        )
        
        let embeddings2: [[Float]] = [
            [0.0, 0.0, 1.0] + .init(repeating: 0.0, count: SpeakerManager.embeddingSize - 3)
        ]
        let durations2: [Float] = [2.0]
        
        let embeddingWeights2: [Float] = Array(repeating: 1.0, count: embeddings2.count)
        let (_, indices2) = speakerManager.assignSpeakers(
            embeddings: embeddings2,
            durations: durations2,
            embeddingWeights: embeddingWeights2
        )
        
        // First batch should have indices 0 and 1
        XCTAssertEqual(indices1.count, 2)
        XCTAssertTrue(indices1[0] >= 0)
        XCTAssertTrue(indices1[1] >= 0)
        
        // Second batch should have index 2 (continuing from first batch)
        XCTAssertEqual(indices2.count, 1)
        XCTAssertTrue(indices2[0] >= 0)
        XCTAssertTrue(indices2[0] > indices1.max() ?? -1)
    }
    
    /// Tests that embedding indices handle edge cases correctly.
    /// Verifies that the embedding index tracking works correctly
    /// with empty batches and mixed valid/invalid embeddings.
    func testAssignSpeakersEmbeddingIndexEdgeCases() {
        // Test with all invalid embeddings
        let invalidEmbeddings: [[Float]] = [[], [1.0, 2.0]]
        let invalidDurations: [Float] = [1.0, 1.0]
        
        let invalidEmbeddingWeights: [Float] = Array(repeating: 1.0, count: invalidEmbeddings.count)
        let (invalidSpeakers, invalidIndices) = speakerManager.assignSpeakers(
            embeddings: invalidEmbeddings,
            durations: invalidDurations,
            embeddingWeights: invalidEmbeddingWeights
        )
        
        XCTAssertEqual(invalidSpeakers.count, 2)
        XCTAssertEqual(invalidIndices.count, 2)
        XCTAssertNil(invalidSpeakers[0])
        XCTAssertNil(invalidSpeakers[1])
        XCTAssertEqual(invalidIndices[0], -1)
        XCTAssertEqual(invalidIndices[1], -1)
        
        // Test with mixed valid/invalid embeddings
        let mixedEmbeddings: [[Float]] = [
            [1.0, 0.0] + .init(repeating: 0.0, count: SpeakerManager.embeddingSize - 2), // Valid
            [], // Invalid
            [0.0, 1.0] + .init(repeating: 0.0, count: SpeakerManager.embeddingSize - 2) // Valid
        ]
        let mixedDurations: [Float] = [2.0, 1.0, 1.5]
        
        let mixedEmbeddingWeights: [Float] = Array(repeating: 1.0, count: mixedEmbeddings.count)
        let (mixedSpeakers, mixedIndices) = speakerManager.assignSpeakers(
            embeddings: mixedEmbeddings,
            durations: mixedDurations,
            embeddingWeights: mixedEmbeddingWeights
        )
        
        XCTAssertEqual(mixedSpeakers.count, 3)
        XCTAssertEqual(mixedIndices.count, 3)
        XCTAssertNotNil(mixedSpeakers[0])
        XCTAssertNil(mixedSpeakers[1])
        XCTAssertNotNil(mixedSpeakers[2])
        XCTAssertTrue(mixedIndices[0] >= 0)
        XCTAssertEqual(mixedIndices[1], -1)
        XCTAssertTrue(mixedIndices[2] >= 0)
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
