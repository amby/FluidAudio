import CoreML
import XCTest
@testable import FluidAudio

/// Comprehensive test suite for EmbeddingExtractor batch processing optimization.
///
/// This test class covers the new batch processing logic that processes up to 3 speakers
/// at once instead of one at a time. The changes include:
/// - Batch processing of multiple speakers in a single model inference
/// - Speaker activity filtering with zero embedding fallback
/// - Optimized memory handling with multiple audio copies
/// - Proper embedding collection and ordering
///
/// Key test categories:
/// - Batch processing with 1-3 active speakers
/// - Mixed active/inactive speaker scenarios
/// - Memory optimization and buffer handling
/// - Edge cases and boundary conditions
/// - Performance improvements
@available(macOS 13.0, iOS 16.0, *)
final class EmbeddingExtractorBatchProcessingTests: XCTestCase {
    
    var embeddingExtractor: EmbeddingExtractor!
    var mockModel: MLModel!
    
    override func setUp() {
        super.setUp()
        // Create a mock MLModel for testing
        // Note: In a real test environment, you would use a proper mock or test model
        do {
            // For now, we'll skip tests that require a real model
            // In a production test environment, you would initialize with a real model
            throw XCTSkip("Cannot test without real MLModel")
        } catch {
            // This will be caught by individual test methods
        }
    }
    
    override func tearDown() {
        embeddingExtractor = nil
        mockModel = nil
        super.tearDown()
    }
    
    // MARK: - Batch Processing Tests
    
    /// Tests batch processing with exactly 3 active speakers.
    /// Verifies that the new batch processing logic correctly handles 3 speakers in one inference.
    func testBatchProcessingWithThreeActiveSpeakers() throws {
        try XCTSkipIf(embeddingExtractor == nil, "Cannot test without real MLModel")
        
        let audio: [Float] = Array(repeating: 0.1, count: 160000) // 10 seconds of audio at 16kHz
        
        // Create masks for 3 active speakers
        let masks: [[Float]] = [
            Array(repeating: 1.0, count: 1000), // Speaker 0: fully active
            Array(repeating: 1.0, count: 1000), // Speaker 1: fully active  
            Array(repeating: 1.0, count: 1000)  // Speaker 2: fully active
        ]
        
        let embeddings = try embeddingExtractor.getEmbeddings(
            audio: audio,
            masks: masks,
            minActivityThreshold: 10.0
        )
        
        XCTAssertEqual(embeddings.count, 3)
        for embedding in embeddings {
            XCTAssertEqual(embedding.count, 256)
            // All embeddings should be non-zero since all speakers are active
            XCTAssertFalse(embedding.allSatisfy { $0 == 0.0 })
        }
    }
    
    /// Tests batch processing with 2 active speakers.
    /// Verifies that the batch processing works correctly with fewer than 3 speakers.
    func testBatchProcessingWithTwoActiveSpeakers() throws {
        try XCTSkipIf(embeddingExtractor == nil, "Cannot test without real MLModel")
        
        let audio: [Float] = Array(repeating: 0.1, count: 160000)
        
        // Create masks for 2 active speakers
        let masks: [[Float]] = [
            Array(repeating: 1.0, count: 1000), // Speaker 0: active
            Array(repeating: 1.0, count: 1000), // Speaker 1: active
            Array(repeating: 0.0, count: 1000)  // Speaker 2: inactive
        ]
        
        let embeddings = try embeddingExtractor.getEmbeddings(
            audio: audio,
            masks: masks,
            minActivityThreshold: 10.0
        )
        
        XCTAssertEqual(embeddings.count, 3)
        // First two embeddings should be non-zero, third should be zero
        XCTAssertFalse(embeddings[0].allSatisfy { $0 == 0.0 })
        XCTAssertFalse(embeddings[1].allSatisfy { $0 == 0.0 })
        XCTAssertTrue(embeddings[2].allSatisfy { $0 == 0.0 })
    }
    
    /// Tests batch processing with 1 active speaker.
    /// Verifies that single speaker processing works correctly in the new batch system.
    func testBatchProcessingWithOneActiveSpeaker() throws {
        try XCTSkipIf(embeddingExtractor == nil, "Cannot test without real MLModel")
        
        let audio: [Float] = Array(repeating: 0.1, count: 160000)
        
        // Create masks for 1 active speaker
        let masks: [[Float]] = [
            Array(repeating: 1.0, count: 1000), // Speaker 0: active
            Array(repeating: 0.0, count: 1000),  // Speaker 1: inactive
            Array(repeating: 0.0, count: 1000)  // Speaker 2: inactive
        ]
        
        let embeddings = try embeddingExtractor.getEmbeddings(
            audio: audio,
            masks: masks,
            minActivityThreshold: 10.0
        )
        
        XCTAssertEqual(embeddings.count, 3)
        // Only first embedding should be non-zero
        XCTAssertFalse(embeddings[0].allSatisfy { $0 == 0.0 })
        XCTAssertTrue(embeddings[1].allSatisfy { $0 == 0.0 })
        XCTAssertTrue(embeddings[2].allSatisfy { $0 == 0.0 })
    }
    
    /// Tests batch processing with all inactive speakers.
    /// Verifies that the system correctly handles cases where no speakers are active.
    func testBatchProcessingWithAllInactiveSpeakers() throws {
        try XCTSkipIf(embeddingExtractor == nil, "Cannot test without real MLModel")
        
        let audio: [Float] = Array(repeating: 0.1, count: 160000)
        
        // Create masks for all inactive speakers
        let masks: [[Float]] = [
            Array(repeating: 0.0, count: 1000), // Speaker 0: inactive
            Array(repeating: 0.0, count: 1000), // Speaker 1: inactive
            Array(repeating: 0.0, count: 1000)  // Speaker 2: inactive
        ]
        
        let embeddings = try embeddingExtractor.getEmbeddings(
            audio: audio,
            masks: masks,
            minActivityThreshold: 10.0
        )
        
        XCTAssertEqual(embeddings.count, 3)
        // All embeddings should be zero
        for embedding in embeddings {
            XCTAssertTrue(embedding.allSatisfy { $0 == 0.0 })
        }
    }
    
    // MARK: - Mixed Active/Inactive Speaker Tests
    
    /// Tests batch processing with mixed active/inactive speakers.
    /// Verifies that the system correctly processes active speakers and fills zeros for inactive ones.
    func testBatchProcessingWithMixedActiveInactiveSpeakers() throws {
        try XCTSkipIf(embeddingExtractor == nil, "Cannot test without real MLModel")
        
        let audio: [Float] = Array(repeating: 0.1, count: 160000)
        
        // Create masks with mixed activity
        let masks: [[Float]] = [
            Array(repeating: 1.0, count: 1000), // Speaker 0: active
            Array(repeating: 0.0, count: 1000), // Speaker 1: inactive
            Array(repeating: 1.0, count: 1000)  // Speaker 2: active
        ]
        
        let embeddings = try embeddingExtractor.getEmbeddings(
            audio: audio,
            masks: masks,
            minActivityThreshold: 10.0
        )
        
        XCTAssertEqual(embeddings.count, 3)
        // Speakers 0 and 2 should be non-zero, speaker 1 should be zero
        XCTAssertFalse(embeddings[0].allSatisfy { $0 == 0.0 })
        XCTAssertTrue(embeddings[1].allSatisfy { $0 == 0.0 })
        XCTAssertFalse(embeddings[2].allSatisfy { $0 == 0.0 })
    }
    
    /// Tests batch processing with more than 3 speakers.
    /// Verifies that the system correctly handles cases with more than 3 speakers by processing in batches.
    func testBatchProcessingWithMoreThanThreeSpeakers() throws {
        try XCTSkipIf(embeddingExtractor == nil, "Cannot test without real MLModel")
        
        let audio: [Float] = Array(repeating: 0.1, count: 160000)
        
        // Create masks for 5 speakers
        let masks: [[Float]] = [
            Array(repeating: 1.0, count: 1000), // Speaker 0: active
            Array(repeating: 0.0, count: 1000),  // Speaker 1: inactive
            Array(repeating: 1.0, count: 1000),  // Speaker 2: active
            Array(repeating: 1.0, count: 1000),  // Speaker 3: active
            Array(repeating: 0.0, count: 1000)   // Speaker 4: inactive
        ]
        
        let embeddings = try embeddingExtractor.getEmbeddings(
            audio: audio,
            masks: masks,
            minActivityThreshold: 10.0
        )
        
        XCTAssertEqual(embeddings.count, 5)
        // Speakers 0, 2, 3 should be non-zero, speakers 1, 4 should be zero
        XCTAssertFalse(embeddings[0].allSatisfy { $0 == 0.0 })
        XCTAssertTrue(embeddings[1].allSatisfy { $0 == 0.0 })
        XCTAssertFalse(embeddings[2].allSatisfy { $0 == 0.0 })
        XCTAssertFalse(embeddings[3].allSatisfy { $0 == 0.0 })
        XCTAssertTrue(embeddings[4].allSatisfy { $0 == 0.0 })
    }
    
    // MARK: - Activity Threshold Tests
    
    /// Tests batch processing with different activity thresholds.
    /// Verifies that the activity threshold correctly filters speakers.
    func testBatchProcessingWithDifferentActivityThresholds() throws {
        try XCTSkipIf(embeddingExtractor == nil, "Cannot test without real MLModel")
        
        let audio: [Float] = Array(repeating: 0.1, count: 160000)
        
        // Create masks with varying activity levels
        let masks: [[Float]] = [
            Array(repeating: 1.0, count: 1000),  // Speaker 0: high activity
            Array(repeating: 0.5, count: 1000),  // Speaker 1: medium activity
            Array(repeating: 0.1, count: 1000)   // Speaker 2: low activity
        ]
        
        // Test with high threshold - only speaker 0 should be active
        let embeddingsHighThreshold = try embeddingExtractor.getEmbeddings(
            audio: audio,
            masks: masks,
            minActivityThreshold: 800.0
        )
        
        XCTAssertEqual(embeddingsHighThreshold.count, 3)
        XCTAssertFalse(embeddingsHighThreshold[0].allSatisfy { $0 == 0.0 })
        XCTAssertTrue(embeddingsHighThreshold[1].allSatisfy { $0 == 0.0 })
        XCTAssertTrue(embeddingsHighThreshold[2].allSatisfy { $0 == 0.0 })
        
        // Test with low threshold - all speakers should be active
        let embeddingsLowThreshold = try embeddingExtractor.getEmbeddings(
            audio: audio,
            masks: masks,
            minActivityThreshold: 50.0
        )
        
        XCTAssertEqual(embeddingsLowThreshold.count, 3)
        XCTAssertFalse(embeddingsLowThreshold[0].allSatisfy { $0 == 0.0 })
        XCTAssertFalse(embeddingsLowThreshold[1].allSatisfy { $0 == 0.0 })
        XCTAssertFalse(embeddingsLowThreshold[2].allSatisfy { $0 == 0.0 })
    }
    
    // MARK: - Memory Optimization Tests
    
    /// Tests that the new memory optimization correctly handles multiple audio copies.
    /// Verifies that the system can handle multiple audio copies for batch processing.
    func testMemoryOptimizationWithMultipleAudioCopies() throws {
        try XCTSkipIf(embeddingExtractor == nil, "Cannot test without real MLModel")
        
        let audio: [Float] = Array(repeating: 0.1, count: 160000)
        
        // Create masks that will trigger multiple audio copies
        let masks: [[Float]] = [
            Array(repeating: 1.0, count: 1000), // Speaker 0: active
            Array(repeating: 1.0, count: 1000), // Speaker 1: active
            Array(repeating: 1.0, count: 1000)   // Speaker 2: active
        ]
        
        // This should trigger the new memory optimization with multiple audio copies
        let embeddings = try embeddingExtractor.getEmbeddings(
            audio: audio,
            masks: masks,
            minActivityThreshold: 10.0
        )
        
        XCTAssertEqual(embeddings.count, 3)
        for embedding in embeddings {
            XCTAssertEqual(embedding.count, 256)
        }
    }
    
    /// Tests that the optimized mask buffer handling works correctly.
    /// Verifies that the new mask buffer optimization correctly handles multiple speakers.
    func testOptimizedMaskBufferHandling() throws {
        try XCTSkipIf(embeddingExtractor == nil, "Cannot test without real MLModel")
        
        let audio: [Float] = Array(repeating: 0.1, count: 160000)
        
        // Create masks with different patterns to test buffer handling
        let masks: [[Float]] = [
            Array(repeating: 1.0, count: 1000), // Speaker 0: all ones
            Array(repeating: 0.0, count: 1000), // Speaker 1: all zeros
            Array(repeating: 0.5, count: 1000)  // Speaker 2: mixed values
        ]
        
        let embeddings = try embeddingExtractor.getEmbeddings(
            audio: audio,
            masks: masks,
            minActivityThreshold: 10.0
        )
        
        XCTAssertEqual(embeddings.count, 3)
        // Results should be consistent with the mask patterns
        XCTAssertFalse(embeddings[0].allSatisfy { $0 == 0.0 })
        XCTAssertTrue(embeddings[1].allSatisfy { $0 == 0.0 })
        XCTAssertFalse(embeddings[2].allSatisfy { $0 == 0.0 })
    }
    
    // MARK: - Edge Cases and Boundary Conditions
    
    /// Tests batch processing with empty masks array.
    /// Verifies that the system handles empty input gracefully.
    func testBatchProcessingWithEmptyMasks() throws {
        try XCTSkipIf(embeddingExtractor == nil, "Cannot test without real MLModel")
        
        let audio: [Float] = Array(repeating: 0.1, count: 160000)
        let masks: [[Float]] = []
        
        let embeddings = try embeddingExtractor.getEmbeddings(
            audio: audio,
            masks: masks,
            minActivityThreshold: 10.0
        )
        
        XCTAssertEqual(embeddings.count, 0)
    }
    
    /// Tests batch processing with single speaker.
    /// Verifies that the system correctly handles single speaker scenarios.
    func testBatchProcessingWithSingleSpeaker() throws {
        try XCTSkipIf(embeddingExtractor == nil, "Cannot test without real MLModel")
        
        let audio: [Float] = Array(repeating: 0.1, count: 160000)
        let masks: [[Float]] = [
            Array(repeating: 1.0, count: 1000) // Single active speaker
        ]
        
        let embeddings = try embeddingExtractor.getEmbeddings(
            audio: audio,
            masks: masks,
            minActivityThreshold: 10.0
        )
        
        XCTAssertEqual(embeddings.count, 1)
        XCTAssertEqual(embeddings[0].count, 256)
        XCTAssertFalse(embeddings[0].allSatisfy { $0 == 0.0 })
    }
    
    /// Tests batch processing with very short audio.
    /// Verifies that the system works with minimal audio duration.
    func testBatchProcessingWithVeryShortAudio() throws {
        try XCTSkipIf(embeddingExtractor == nil, "Cannot test without real MLModel")
        
        let audio: [Float] = Array(repeating: 0.1, count: 1600) // 0.1 seconds at 16kHz
        let masks: [[Float]] = [
            Array(repeating: 1.0, count: 100),
            Array(repeating: 1.0, count: 100),
            Array(repeating: 1.0, count: 100)
        ]
        
        let embeddings = try embeddingExtractor.getEmbeddings(
            audio: audio,
            masks: masks,
            minActivityThreshold: 10.0
        )
        
        XCTAssertEqual(embeddings.count, 3)
        for embedding in embeddings {
            XCTAssertEqual(embedding.count, 256)
        }
    }
    
    /// Tests batch processing with mismatched mask sizes.
    /// Verifies that the system handles masks of different sizes correctly.
    func testBatchProcessingWithMismatchedMaskSizes() throws {
        try XCTSkipIf(embeddingExtractor == nil, "Cannot test without real MLModel")
        
        let audio: [Float] = Array(repeating: 0.1, count: 160000)
        let masks: [[Float]] = [
            Array(repeating: 1.0, count: 1000), // Speaker 0: 1000 frames
            Array(repeating: 1.0, count: 500),  // Speaker 1: 500 frames
            Array(repeating: 1.0, count: 1500) // Speaker 2: 1500 frames
        ]
        
        let embeddings = try embeddingExtractor.getEmbeddings(
            audio: audio,
            masks: masks,
            minActivityThreshold: 10.0
        )
        
        XCTAssertEqual(embeddings.count, 3)
        for embedding in embeddings {
            XCTAssertEqual(embedding.count, 256)
        }
    }
    
    // MARK: - Performance Tests
    
    /// Tests performance improvement with batch processing.
    /// Verifies that the new batch processing is more efficient than sequential processing.
    func testBatchProcessingPerformanceImprovement() throws {
        try XCTSkipIf(embeddingExtractor == nil, "Cannot test without real MLModel")
        
        let audio: [Float] = Array(repeating: 0.1, count: 160000)
        let masks: [[Float]] = [
            Array(repeating: 1.0, count: 1000),
            Array(repeating: 1.0, count: 1000),
            Array(repeating: 1.0, count: 1000)
        ]
        
        measure {
            do {
                let embeddings = try embeddingExtractor.getEmbeddings(
                    audio: audio,
                    masks: masks,
                    minActivityThreshold: 10.0
                )
                XCTAssertEqual(embeddings.count, 3)
            } catch {
                XCTFail("Performance test failed: \(error)")
            }
        }
    }
    
    /// Tests performance with many speakers (more than 3).
    /// Verifies that the system efficiently handles large numbers of speakers.
    func testBatchProcessingPerformanceWithManySpeakers() throws {
        try XCTSkipIf(embeddingExtractor == nil, "Cannot test without real MLModel")
        
        let audio: [Float] = Array(repeating: 0.1, count: 160000)
        
        // Create masks for 10 speakers
        let masks: [[Float]] = (0..<10).map { i in
            Array(repeating: i % 2 == 0 ? 1.0 : 0.0, count: 1000)
        }
        
        measure {
            do {
                let embeddings = try embeddingExtractor.getEmbeddings(
                    audio: audio,
                    masks: masks,
                    minActivityThreshold: 10.0
                )
                XCTAssertEqual(embeddings.count, 10)
            } catch {
                XCTFail("Performance test failed: \(error)")
            }
        }
    }
    
    // MARK: - Consistency Tests
    
    /// Tests that batch processing produces consistent results.
    /// Verifies that multiple runs with the same input produce the same results.
    func testBatchProcessingConsistency() throws {
        try XCTSkipIf(embeddingExtractor == nil, "Cannot test without real MLModel")
        
        let audio: [Float] = Array(repeating: 0.1, count: 160000)
        let masks: [[Float]] = [
            Array(repeating: 1.0, count: 1000),
            Array(repeating: 0.0, count: 1000),
            Array(repeating: 1.0, count: 1000)
        ]
        
        // Run multiple times and compare results
        let embeddings1 = try embeddingExtractor.getEmbeddings(
            audio: audio,
            masks: masks,
            minActivityThreshold: 10.0
        )
        
        let embeddings2 = try embeddingExtractor.getEmbeddings(
            audio: audio,
            masks: masks,
            minActivityThreshold: 10.0
        )
        
        XCTAssertEqual(embeddings1.count, embeddings2.count)
        
        for (embedding1, embedding2) in zip(embeddings1, embeddings2) {
            XCTAssertEqual(embedding1.count, embedding2.count)
            
            // Check that embeddings are identical (within floating-point precision)
            let differences = zip(embedding1, embedding2).map { abs($0 - $1) }
            let maxDifference = differences.max() ?? 0
            XCTAssertLessThan(maxDifference, 0.0001, "Embeddings should be identical")
        }
    }
    
    // MARK: - Helper Methods
    
    /// Generates test audio data for testing purposes.
    /// Creates audio data with a specific pattern to ensure reproducible test results.
    private func generateTestAudio(duration: Float) -> [Float] {
        let sampleCount = Int(duration * 16000) // 16000 samples per second (16kHz)
        return (0..<sampleCount).map { i in
            // Create a simple sine wave pattern
            sin(Float(i) * 0.01) * 0.1
        }
    }
    
    /// Generates test masks for testing purposes.
    /// Creates masks with specific patterns to test various scenarios.
    private func generateTestMasks(speakerCount: Int, frameCount: Int, activeSpeakers: Set<Int>) -> [[Float]] {
        return (0..<speakerCount).map { speakerIndex in
            if activeSpeakers.contains(speakerIndex) {
                return Array(repeating: 1.0, count: frameCount)
            } else {
                return Array(repeating: 0.0, count: frameCount)
            }
        }
    }
}
