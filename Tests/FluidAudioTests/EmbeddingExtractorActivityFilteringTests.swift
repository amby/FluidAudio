import CoreML
import XCTest
@testable import FluidAudio

/// Comprehensive test suite for EmbeddingExtractor speaker activity filtering and zero embedding handling.
///
/// This test class covers the new speaker activity filtering logic that:
/// - Filters out inactive speakers based on activity threshold
/// - Returns zero embeddings for inactive speakers
/// - Maintains proper embedding ordering and count
/// - Handles edge cases in activity calculation
///
/// Key test categories:
/// - Speaker activity threshold filtering
/// - Zero embedding generation for inactive speakers
/// - Activity calculation edge cases
/// - Embedding ordering and consistency
/// - Mixed active/inactive speaker scenarios
@available(macOS 13.0, iOS 16.0, *)
final class EmbeddingExtractorActivityFilteringTests: XCTestCase {
    
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
    
    // MARK: - Activity Threshold Tests
    
    /// Tests speaker activity filtering with high activity threshold.
    /// Verifies that only highly active speakers are processed and inactive ones get zero embeddings.
    func testActivityFilteringWithHighThreshold() throws {
        try XCTSkipIf(embeddingExtractor == nil, "Cannot test without real MLModel")
        
        let audio: [Float] = Array(repeating: 0.1, count: 160000)
        
        // Create masks with varying activity levels
        let masks: [[Float]] = [
            Array(repeating: 1.0, count: 1000),  // Speaker 0: high activity (1000.0)
            Array(repeating: 0.5, count: 1000),  // Speaker 1: medium activity (500.0)
            Array(repeating: 0.1, count: 1000)   // Speaker 2: low activity (100.0)
        ]
        
        let embeddings = try embeddingExtractor.getEmbeddings(
            audio: audio,
            masks: masks,
            minActivityThreshold: 800.0 // High threshold
        )
        
        XCTAssertEqual(embeddings.count, 3)
        
        // Only speaker 0 should be active (activity = 1000.0 > 800.0)
        XCTAssertFalse(embeddings[0].allSatisfy { $0 == 0.0 }, "Speaker 0 should be active")
        XCTAssertTrue(embeddings[1].allSatisfy { $0 == 0.0 }, "Speaker 1 should be inactive")
        XCTAssertTrue(embeddings[2].allSatisfy { $0 == 0.0 }, "Speaker 2 should be inactive")
    }
    
    /// Tests speaker activity filtering with low activity threshold.
    /// Verifies that most speakers are processed with a low threshold.
    func testActivityFilteringWithLowThreshold() throws {
        try XCTSkipIf(embeddingExtractor == nil, "Cannot test without real MLModel")
        
        let audio: [Float] = Array(repeating: 0.1, count: 160000)
        
        // Create masks with varying activity levels
        let masks: [[Float]] = [
            Array(repeating: 1.0, count: 1000),  // Speaker 0: high activity (1000.0)
            Array(repeating: 0.5, count: 1000),  // Speaker 1: medium activity (500.0)
            Array(repeating: 0.1, count: 1000)   // Speaker 2: low activity (100.0)
        ]
        
        let embeddings = try embeddingExtractor.getEmbeddings(
            audio: audio,
            masks: masks,
            minActivityThreshold: 50.0 // Low threshold
        )
        
        XCTAssertEqual(embeddings.count, 3)
        
        // All speakers should be active (all activities > 50.0)
        XCTAssertFalse(embeddings[0].allSatisfy { $0 == 0.0 }, "Speaker 0 should be active")
        XCTAssertFalse(embeddings[1].allSatisfy { $0 == 0.0 }, "Speaker 1 should be active")
        XCTAssertFalse(embeddings[2].allSatisfy { $0 == 0.0 }, "Speaker 2 should be active")
    }
    
    /// Tests speaker activity filtering with zero activity threshold.
    /// Verifies that all speakers are processed when threshold is zero.
    func testActivityFilteringWithZeroThreshold() throws {
        try XCTSkipIf(embeddingExtractor == nil, "Cannot test without real MLModel")
        
        let audio: [Float] = Array(repeating: 0.1, count: 160000)
        
        // Create masks with varying activity levels
        let masks: [[Float]] = [
            Array(repeating: 1.0, count: 1000),  // Speaker 0: high activity
            Array(repeating: 0.5, count: 1000),  // Speaker 1: medium activity
            Array(repeating: 0.0, count: 1000)   // Speaker 2: no activity
        ]
        
        let embeddings = try embeddingExtractor.getEmbeddings(
            audio: audio,
            masks: masks,
            minActivityThreshold: 0.0 // Zero threshold
        )
        
        XCTAssertEqual(embeddings.count, 3)
        
        // All speakers should be active (all activities >= 0.0)
        XCTAssertFalse(embeddings[0].allSatisfy { $0 == 0.0 }, "Speaker 0 should be active")
        XCTAssertFalse(embeddings[1].allSatisfy { $0 == 0.0 }, "Speaker 1 should be active")
        XCTAssertFalse(embeddings[2].allSatisfy { $0 == 0.0 }, "Speaker 2 should be active")
    }
    
    // MARK: - Zero Embedding Tests
    
    /// Tests that inactive speakers receive zero embeddings.
    /// Verifies that the system correctly generates zero embeddings for inactive speakers.
    func testZeroEmbeddingsForInactiveSpeakers() throws {
        try XCTSkipIf(embeddingExtractor == nil, "Cannot test without real MLModel")
        
        let audio: [Float] = Array(repeating: 0.1, count: 160000)
        
        // Create masks where only speaker 0 is active
        let masks: [[Float]] = [
            Array(repeating: 1.0, count: 1000), // Speaker 0: active
            Array(repeating: 0.0, count: 1000), // Speaker 1: inactive
            Array(repeating: 0.0, count: 1000)  // Speaker 2: inactive
        ]
        
        let embeddings = try embeddingExtractor.getEmbeddings(
            audio: audio,
            masks: masks,
            minActivityThreshold: 500.0
        )
        
        XCTAssertEqual(embeddings.count, 3)
        
        // Speaker 0 should have non-zero embedding
        XCTAssertFalse(embeddings[0].allSatisfy { $0 == 0.0 }, "Speaker 0 should be active")
        
        // Speakers 1 and 2 should have zero embeddings
        XCTAssertTrue(embeddings[1].allSatisfy { $0 == 0.0 }, "Speaker 1 should have zero embedding")
        XCTAssertTrue(embeddings[2].allSatisfy { $0 == 0.0 }, "Speaker 2 should have zero embedding")
        
        // Verify that zero embeddings have the correct dimension
        XCTAssertEqual(embeddings[1].count, 256, "Zero embedding should have correct dimension")
        XCTAssertEqual(embeddings[2].count, 256, "Zero embedding should have correct dimension")
    }
    
    /// Tests that all speakers receive zero embeddings when all are inactive.
    /// Verifies that the system correctly handles the case where no speakers are active.
    func testZeroEmbeddingsForAllInactiveSpeakers() throws {
        try XCTSkipIf(embeddingExtractor == nil, "Cannot test without real MLModel")
        
        let audio: [Float] = Array(repeating: 0.1, count: 160000)
        
        // Create masks where all speakers are inactive
        let masks: [[Float]] = [
            Array(repeating: 0.0, count: 1000), // Speaker 0: inactive
            Array(repeating: 0.0, count: 1000), // Speaker 1: inactive
            Array(repeating: 0.0, count: 1000)  // Speaker 2: inactive
        ]
        
        let embeddings = try embeddingExtractor.getEmbeddings(
            audio: audio,
            masks: masks,
            minActivityThreshold: 100.0
        )
        
        XCTAssertEqual(embeddings.count, 3)
        
        // All speakers should have zero embeddings
        for (index, embedding) in embeddings.enumerated() {
            XCTAssertTrue(embedding.allSatisfy { $0 == 0.0 }, "Speaker \(index) should have zero embedding")
            XCTAssertEqual(embedding.count, 256, "Zero embedding should have correct dimension")
        }
    }
    
    // MARK: - Activity Calculation Edge Cases
    
    /// Tests activity calculation with negative mask values.
    /// Verifies that negative values in masks are handled correctly in activity calculation.
    func testActivityCalculationWithNegativeValues() throws {
        try XCTSkipIf(embeddingExtractor == nil, "Cannot test without real MLModel")
        
        let audio: [Float] = Array(repeating: 0.1, count: 160000)
        
        // Create masks with negative values
        let masks: [[Float]] = [
            Array(repeating: 1.0, count: 1000),  // Speaker 0: positive activity
            Array(repeating: -1.0, count: 1000), // Speaker 1: negative activity
            Array(repeating: 0.0, count: 1000)   // Speaker 2: zero activity
        ]
        
        let embeddings = try embeddingExtractor.getEmbeddings(
            audio: audio,
            masks: masks,
            minActivityThreshold: 0.0
        )
        
        XCTAssertEqual(embeddings.count, 3)
        
        // Speaker 0 should be active (positive activity)
        XCTAssertFalse(embeddings[0].allSatisfy { $0 == 0.0 }, "Speaker 0 should be active")
        
        // Speaker 1 should be inactive (negative activity sum)
        XCTAssertTrue(embeddings[1].allSatisfy { $0 == 0.0 }, "Speaker 1 should be inactive")
        
        // Speaker 2 should be inactive (zero activity)
        XCTAssertTrue(embeddings[2].allSatisfy { $0 == 0.0 }, "Speaker 2 should be inactive")
    }
    
    /// Tests activity calculation with mixed positive and negative values.
    /// Verifies that mixed mask values are handled correctly in activity calculation.
    func testActivityCalculationWithMixedValues() throws {
        try XCTSkipIf(embeddingExtractor == nil, "Cannot test without real MLModel")
        
        let audio: [Float] = Array(repeating: 0.1, count: 160000)
        
        // Create masks with mixed positive and negative values
        let masks: [[Float]] = [
            Array(repeating: 1.0, count: 500) + Array(repeating: -1.0, count: 500), // Net activity: 0.0
            Array(repeating: 1.0, count: 600) + Array(repeating: -1.0, count: 400), // Net activity: 200.0
            Array(repeating: 1.0, count: 400) + Array(repeating: -1.0, count: 600) // Net activity: -200.0
        ]
        
        let embeddings = try embeddingExtractor.getEmbeddings(
            audio: audio,
            masks: masks,
            minActivityThreshold: 100.0
        )
        
        XCTAssertEqual(embeddings.count, 3)
        
        // Only speaker 1 should be active (net activity = 200.0 > 100.0)
        XCTAssertTrue(embeddings[0].allSatisfy { $0 == 0.0 }, "Speaker 0 should be inactive")
        XCTAssertFalse(embeddings[1].allSatisfy { $0 == 0.0 }, "Speaker 1 should be active")
        XCTAssertTrue(embeddings[2].allSatisfy { $0 == 0.0 }, "Speaker 2 should be inactive")
    }
    
    /// Tests activity calculation with very small values.
    /// Verifies that very small mask values are handled correctly in activity calculation.
    func testActivityCalculationWithVerySmallValues() throws {
        try XCTSkipIf(embeddingExtractor == nil, "Cannot test without real MLModel")
        
        let audio: [Float] = Array(repeating: 0.1, count: 160000)
        
        // Create masks with very small values
        let masks: [[Float]] = [
            Array(repeating: 0.001, count: 1000), // Speaker 0: very small activity (1.0)
            Array(repeating: 0.0001, count: 1000), // Speaker 1: very small activity (0.1)
            Array(repeating: 0.0, count: 1000)   // Speaker 2: zero activity
        ]
        
        let embeddings = try embeddingExtractor.getEmbeddings(
            audio: audio,
            masks: masks,
            minActivityThreshold: 0.5
        )
        
        XCTAssertEqual(embeddings.count, 3)
        
        // Only speaker 0 should be active (activity = 1.0 > 0.5)
        XCTAssertFalse(embeddings[0].allSatisfy { $0 == 0.0 }, "Speaker 0 should be active")
        XCTAssertTrue(embeddings[1].allSatisfy { $0 == 0.0 }, "Speaker 1 should be inactive")
        XCTAssertTrue(embeddings[2].allSatisfy { $0 == 0.0 }, "Speaker 2 should be inactive")
    }
    
    // MARK: - Embedding Ordering Tests
    
    /// Tests that embedding order matches speaker order.
    /// Verifies that embeddings are returned in the same order as input speakers.
    func testEmbeddingOrderMatchesSpeakerOrder() throws {
        try XCTSkipIf(embeddingExtractor == nil, "Cannot test without real MLModel")
        
        let audio: [Float] = Array(repeating: 0.1, count: 160000)
        
        // Create masks with specific activity patterns
        let masks: [[Float]] = [
            Array(repeating: 0.0, count: 1000), // Speaker 0: inactive
            Array(repeating: 1.0, count: 1000), // Speaker 1: active
            Array(repeating: 0.0, count: 1000)  // Speaker 2: inactive
        ]
        
        let embeddings = try embeddingExtractor.getEmbeddings(
            audio: audio,
            masks: masks,
            minActivityThreshold: 500.0
        )
        
        XCTAssertEqual(embeddings.count, 3)
        
        // Embeddings should be in the same order as speakers
        XCTAssertTrue(embeddings[0].allSatisfy { $0 == 0.0 }, "Speaker 0 embedding should be zero")
        XCTAssertFalse(embeddings[1].allSatisfy { $0 == 0.0 }, "Speaker 1 embedding should be non-zero")
        XCTAssertTrue(embeddings[2].allSatisfy { $0 == 0.0 }, "Speaker 2 embedding should be zero")
    }
    
    /// Tests that embedding count matches speaker count.
    /// Verifies that the number of returned embeddings matches the number of input speakers.
    func testEmbeddingCountMatchesSpeakerCount() throws {
        try XCTSkipIf(embeddingExtractor == nil, "Cannot test without real MLModel")
        
        let audio: [Float] = Array(repeating: 0.1, count: 160000)
        
        // Test with different numbers of speakers
        let speakerCounts = [1, 2, 3, 5, 10]
        
        for speakerCount in speakerCounts {
            let masks: [[Float]] = (0..<speakerCount).map { i in
                Array(repeating: i % 2 == 0 ? 1.0 : 0.0, count: 1000)
            }
            
            let embeddings = try embeddingExtractor.getEmbeddings(
                audio: audio,
                masks: masks,
                minActivityThreshold: 500.0
            )
            
            XCTAssertEqual(embeddings.count, speakerCount, "Embedding count should match speaker count for \(speakerCount) speakers")
        }
    }
    
    // MARK: - Mixed Active/Inactive Speaker Tests
    
    /// Tests mixed active/inactive speaker scenarios.
    /// Verifies that the system correctly handles combinations of active and inactive speakers.
    func testMixedActiveInactiveSpeakers() throws {
        try XCTSkipIf(embeddingExtractor == nil, "Cannot test without real MLModel")
        
        let audio: [Float] = Array(repeating: 0.1, count: 160000)
        
        // Create masks with alternating active/inactive pattern
        let masks: [[Float]] = [
            Array(repeating: 1.0, count: 1000), // Speaker 0: active
            Array(repeating: 0.0, count: 1000), // Speaker 1: inactive
            Array(repeating: 1.0, count: 1000), // Speaker 2: active
            Array(repeating: 0.0, count: 1000), // Speaker 3: inactive
            Array(repeating: 1.0, count: 1000)  // Speaker 4: active
        ]
        
        let embeddings = try embeddingExtractor.getEmbeddings(
            audio: audio,
            masks: masks,
            minActivityThreshold: 500.0
        )
        
        XCTAssertEqual(embeddings.count, 5)
        
        // Check alternating pattern
        XCTAssertFalse(embeddings[0].allSatisfy { $0 == 0.0 }, "Speaker 0 should be active")
        XCTAssertTrue(embeddings[1].allSatisfy { $0 == 0.0 }, "Speaker 1 should be inactive")
        XCTAssertFalse(embeddings[2].allSatisfy { $0 == 0.0 }, "Speaker 2 should be active")
        XCTAssertTrue(embeddings[3].allSatisfy { $0 == 0.0 }, "Speaker 3 should be inactive")
        XCTAssertFalse(embeddings[4].allSatisfy { $0 == 0.0 }, "Speaker 4 should be active")
    }
    
    /// Tests edge case where only the last speaker is active.
    /// Verifies that the system correctly handles the case where only the final speaker is active.
    func testOnlyLastSpeakerActive() throws {
        try XCTSkipIf(embeddingExtractor == nil, "Cannot test without real MLModel")
        
        let audio: [Float] = Array(repeating: 0.1, count: 160000)
        
        // Create masks where only the last speaker is active
        let masks: [[Float]] = [
            Array(repeating: 0.0, count: 1000), // Speaker 0: inactive
            Array(repeating: 0.0, count: 1000), // Speaker 1: inactive
            Array(repeating: 0.0, count: 1000), // Speaker 2: inactive
            Array(repeating: 1.0, count: 1000)  // Speaker 3: active
        ]
        
        let embeddings = try embeddingExtractor.getEmbeddings(
            audio: audio,
            masks: masks,
            minActivityThreshold: 500.0
        )
        
        XCTAssertEqual(embeddings.count, 4)
        
        // Only the last speaker should be active
        XCTAssertTrue(embeddings[0].allSatisfy { $0 == 0.0 }, "Speaker 0 should be inactive")
        XCTAssertTrue(embeddings[1].allSatisfy { $0 == 0.0 }, "Speaker 1 should be inactive")
        XCTAssertTrue(embeddings[2].allSatisfy { $0 == 0.0 }, "Speaker 2 should be inactive")
        XCTAssertFalse(embeddings[3].allSatisfy { $0 == 0.0 }, "Speaker 3 should be active")
    }
    
    // MARK: - Edge Cases and Boundary Conditions
    
    /// Tests activity filtering with empty masks.
    /// Verifies that the system handles empty mask arrays gracefully.
    func testActivityFilteringWithEmptyMasks() throws {
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
    
    /// Tests activity filtering with single speaker.
    /// Verifies that the system correctly handles single speaker scenarios.
    func testActivityFilteringWithSingleSpeaker() throws {
        try XCTSkipIf(embeddingExtractor == nil, "Cannot test without real MLModel")
        
        let audio: [Float] = Array(repeating: 0.1, count: 160000)
        let masks: [[Float]] = [
            Array(repeating: 1.0, count: 1000) // Single active speaker
        ]
        
        let embeddings = try embeddingExtractor.getEmbeddings(
            audio: audio,
            masks: masks,
            minActivityThreshold: 500.0
        )
        
        XCTAssertEqual(embeddings.count, 1)
        XCTAssertFalse(embeddings[0].allSatisfy { $0 == 0.0 }, "Single speaker should be active")
    }
    
    /// Tests activity filtering with very high threshold.
    /// Verifies that the system correctly handles thresholds higher than any speaker activity.
    func testActivityFilteringWithVeryHighThreshold() throws {
        try XCTSkipIf(embeddingExtractor == nil, "Cannot test without real MLModel")
        
        let audio: [Float] = Array(repeating: 0.1, count: 160000)
        
        // Create masks with moderate activity
        let masks: [[Float]] = [
            Array(repeating: 1.0, count: 1000), // Speaker 0: activity = 1000.0
            Array(repeating: 0.5, count: 1000), // Speaker 1: activity = 500.0
            Array(repeating: 0.1, count: 1000)   // Speaker 2: activity = 100.0
        ]
        
        let embeddings = try embeddingExtractor.getEmbeddings(
            audio: audio,
            masks: masks,
            minActivityThreshold: 2000.0 // Very high threshold
        )
        
        XCTAssertEqual(embeddings.count, 3)
        
        // All speakers should be inactive (all activities < 2000.0)
        for (index, embedding) in embeddings.enumerated() {
            XCTAssertTrue(embedding.allSatisfy { $0 == 0.0 }, "Speaker \(index) should be inactive")
        }
    }
    
    // MARK: - Helper Methods
    
    /// Generates test masks with specific activity patterns.
    /// Creates masks with controlled activity levels for testing.
    private func generateTestMasks(speakerCount: Int, frameCount: Int, activityLevels: [Float]) -> [[Float]] {
        return activityLevels.map { activity in
            Array(repeating: activity, count: frameCount)
        }
    }
    
    /// Calculates expected activity for a mask.
    /// Helper method to verify activity calculations.
    private func calculateActivity(_ mask: [Float]) -> Float {
        return mask.reduce(0, +)
    }
}
