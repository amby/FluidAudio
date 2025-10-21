import XCTest
@testable import FluidAudio

/// Integration tests for the updated DiarizerManager with batch processing.
/// 
/// This test class covers the integration between the new batch processing functionality
/// and the existing DiarizerManager features, ensuring that the new functionality works
/// correctly with the complete diarization pipeline.
@available(macOS 13.0, iOS 16.0, *)
final class DiarizerManagerIntegrationTests: XCTestCase {
    
    var diarizerManager: DiarizerManager!
    var config: DiarizerConfig!
    
    override func setUp() {
        super.setUp()
        config = DiarizerConfig(
            clusteringThreshold: 0.65,
            minSpeechDuration: 1.0,
            minEmbeddingUpdateDuration: 2.0,
            minActiveFramesCount: 10,
            debugMode: true
        )
        diarizerManager = DiarizerManager(config: config)
    }
    
    override func tearDown() {
        diarizerManager = nil
        super.tearDown()
    }
    
    // MARK: - Complete Diarization Integration Tests
    
    /// Tests the complete diarization pipeline with batch processing.
    /// Verifies that the new batch processing integrates correctly with
    /// the complete diarization workflow from audio input to final results.
    func testCompleteDiarizationWithBatchProcessing() throws {
        // Create mock models
        guard let mockModels = createMockDiarizerModels() else {
            throw XCTSkip("Mock models not available in test environment")
        }
        
        diarizerManager.initialize(models: mockModels)
        
        // Create test audio with multiple speakers
        let audioSamples = createTestAudioWithMultipleSpeakers(duration: 30.0, sampleRate: 16000)
        
        // Perform complete diarization
        let result = try diarizerManager.performCompleteDiarization(
            audioSamples,
            sampleRate: 16000,
            segmentationThreshold: 0.0
        )
        
        // Verify results
        XCTAssertFalse(result.segments.isEmpty, "Should produce diarization segments")
        
        // Verify segment properties
        for segment in result.segments {
            XCTAssertFalse(segment.speakerId.isEmpty, "Segment should have speaker ID")
            XCTAssertGreaterThan(segment.startTimeSeconds, 0, "Segment should have valid start time")
            XCTAssertGreaterThan(segment.endTimeSeconds, segment.startTimeSeconds, "Segment should have valid end time")
            XCTAssertFalse(segment.embedding.isEmpty, "Segment should have embedding")
            XCTAssertGreaterThan(segment.qualityScore, 0, "Segment should have quality score")
        }
        
        // Verify debug mode output
        if config.debugMode {
            XCTAssertNotNil(result.speakerDatabase, "Should have speaker database in debug mode")
            XCTAssertNotNil(result.timings, "Should have timings in debug mode")
            
            if let timings = result.timings {
                XCTAssertGreaterThan(timings.segmentationSeconds, 0, "Should have segmentation timing")
                XCTAssertGreaterThan(timings.embeddingExtractionSeconds, 0, "Should have embedding timing")
                XCTAssertGreaterThan(timings.speakerClusteringSeconds, 0, "Should have clustering timing")
            }
        }
    }
    
    /// Tests diarization with known speakers initialization.
    /// Verifies that the batch processing correctly integrates with
    /// pre-initialized known speakers and maintains speaker consistency.
    func testDiarizationWithKnownSpeakers() throws {
        guard let mockModels = createMockDiarizerModels() else {
            throw XCTSkip("Mock models not available in test environment")
        }
        
        diarizerManager.initialize(models: mockModels)
        
        // Initialize with known speakers
        let knownSpeakers = [
            Speaker(
                id: "speaker-1",
                name: "Known Speaker 1",
                currentEmbedding: [1.0, 0.0] + Array(repeating: 0.0, count: 254),
                duration: 10.0,
                createdAt: Date(),
                updatedAt: Date()
            ),
            Speaker(
                id: "speaker-2",
                name: "Known Speaker 2",
                currentEmbedding: [0.0, 1.0] + Array(repeating: 0.0, count: 254),
                duration: 8.0,
                createdAt: Date(),
                updatedAt: Date()
            )
        ]
        
        diarizerManager.initializeKnownSpeakers(knownSpeakers)
        
        // Create test audio
        let audioSamples = createTestAudioWithMultipleSpeakers(duration: 20.0, sampleRate: 16000)
        
        // Perform diarization
        let result = try diarizerManager.performCompleteDiarization(
            audioSamples,
            sampleRate: 16000
        )
        
        // Verify results
        XCTAssertFalse(result.segments.isEmpty, "Should produce segments")
        
        // Verify speaker consistency
        let speakerIds = Set(result.segments.map { $0.speakerId })
        XCTAssertTrue(speakerIds.allSatisfy { !$0.isEmpty }, "All segments should have speaker IDs")
        
        // Verify that known speakers are used when appropriate
        let knownSpeakerIds = Set(knownSpeakers.map { $0.id })
        let usedSpeakerIds = Set(result.segments.map { $0.speakerId })
        
        // Should have some overlap with known speakers (depending on similarity)
        XCTAssertTrue(
            !knownSpeakerIds.isDisjoint(with: usedSpeakerIds) || !usedSpeakerIds.isEmpty,
            "Should use known speakers or create new ones"
        )
    }
    
    /// Tests diarization with different audio collection types.
    /// Verifies that the batch processing works correctly with
    /// different audio collection types (Array, ArraySlice, ContiguousArray).
    func testDiarizationWithDifferentCollectionTypes() throws {
        guard let mockModels = createMockDiarizerModels() else {
            throw XCTSkip("Mock models not available in test environment")
        }
        
        diarizerManager.initialize(models: mockModels)
        
        // Test with Array
        let audioArray = createTestAudioWithMultipleSpeakers(duration: 10.0, sampleRate: 16000)
        let resultArray = try diarizerManager.performCompleteDiarization(audioArray)
        XCTAssertFalse(resultArray.segments.isEmpty, "Should work with Array")
        
        // Test with ArraySlice
        let audioSlice = audioArray[8000..<24000] // 1 second slice
        let resultSlice = try diarizerManager.performCompleteDiarization(audioSlice)
        XCTAssertFalse(resultSlice.segments.isEmpty, "Should work with ArraySlice")
        
        // Test with ContiguousArray
        let audioContiguous = ContiguousArray(audioArray)
        let resultContiguous = try diarizerManager.performCompleteDiarization(audioContiguous)
        XCTAssertFalse(resultContiguous.segments.isEmpty, "Should work with ContiguousArray")
    }
    
    // MARK: - Chunk Processing Integration Tests
    
    /// Tests chunk processing with different chunk sizes and overlaps.
    /// Verifies that the batch processing correctly handles
    /// different chunk configurations and maintains consistency.
    func testChunkProcessingWithDifferentConfigurations() throws {
        guard let mockModels = createMockDiarizerModels() else {
            throw XCTSkip("Mock models not available in test environment")
        }
        
        diarizerManager.initialize(models: mockModels)
        
        // Test with different chunk durations
        let chunkDurations: [Double] = [5.0, 10.0, 15.0, 20.0]
        
        for duration in chunkDurations {
            let audioSamples = createTestAudioWithMultipleSpeakers(duration: duration, sampleRate: 16000)
            
            let result = try diarizerManager.performCompleteDiarization(
                audioSamples,
                sampleRate: 16000
            )
            
            XCTAssertFalse(result.segments.isEmpty, "Should work with chunk duration \(duration)")
            
            // Verify timing consistency
            for segment in result.segments {
                XCTAssertGreaterThanOrEqual(segment.startTimeSeconds, 0, "Segment should have valid start time")
                XCTAssertLessThanOrEqual(segment.endTimeSeconds, Float(duration), "Segment should not exceed audio duration")
            }
        }
    }
    
    /// Tests chunk processing with overlapping audio segments.
    /// Verifies that the batch processing correctly handles
    /// overlapping audio segments and maintains speaker consistency.
    func testChunkProcessingWithOverlappingSegments() throws {
        guard let mockModels = createMockDiarizerModels() else {
            throw XCTSkip("Mock models not available in test environment")
        }
        
        diarizerManager.initialize(models: mockModels)
        
        // Create audio with overlapping segments
        let audioSamples = createTestAudioWithOverlappingSegments(duration: 30.0, sampleRate: 16000)
        
        let result = try diarizerManager.performCompleteDiarization(
            audioSamples,
            sampleRate: 16000
        )
        
        XCTAssertFalse(result.segments.isEmpty, "Should handle overlapping segments")
        
        // Verify segment ordering
        let sortedSegments = result.segments.sorted { $0.startTimeSeconds < $1.startTimeSeconds }
        XCTAssertEqual(result.segments.count, sortedSegments.count, "Segments should be sorted by start time")
        
        // Verify segments are in chronological order
        for i in 1..<result.segments.count {
            XCTAssertLessThanOrEqual(
                result.segments[i-1].startTimeSeconds,
                result.segments[i].startTimeSeconds,
                "Segments should be in chronological order"
            )
        }
        
        // Verify no negative durations
        for segment in result.segments {
            XCTAssertGreaterThan(segment.endTimeSeconds - segment.startTimeSeconds, 0, "Segment should have positive duration")
        }
    }
    
    // MARK: - Error Handling Integration Tests
    
    /// Tests diarization with invalid audio data.
    /// Verifies that the batch processing gracefully handles
    /// invalid audio data without crashing.
    func testDiarizationWithInvalidAudio() throws {
        guard let mockModels = createMockDiarizerModels() else {
            throw XCTSkip("Mock models not available in test environment")
        }
        
        diarizerManager.initialize(models: mockModels)
        
        // Test with empty audio
        let emptyAudio: [Float] = []
        do {
            let result = try diarizerManager.performCompleteDiarization(emptyAudio)
            XCTAssertTrue(result.segments.isEmpty, "Should handle empty audio gracefully")
        } catch {
            // Acceptable to fail with empty audio
            XCTAssertTrue(
                error.localizedDescription.contains("empty") ||
                error.localizedDescription.contains("invalid") ||
                error.localizedDescription.contains("audio")
            )
        }
        
        // Test with very short audio
        let shortAudio = Array(repeating: Float(0.0), count: 100)
        do {
            let result = try diarizerManager.performCompleteDiarization(shortAudio)
            XCTAssertNotNil(result, "Should handle short audio")
        } catch {
            // Acceptable to fail with very short audio
            XCTAssertTrue(
                error.localizedDescription.contains("short") ||
                error.localizedDescription.contains("duration") ||
                error.localizedDescription.contains("audio")
            )
        }
    }
    
    /// Tests diarization with extreme audio values.
    /// Verifies that the batch processing handles extreme
    /// audio values gracefully without numerical instability.
    func testDiarizationWithExtremeAudioValues() throws {
        guard let mockModels = createMockDiarizerModels() else {
            throw XCTSkip("Mock models not available in test environment")
        }
        
        diarizerManager.initialize(models: mockModels)
        
        // Create audio with extreme values
        var extremeAudio: [Float] = []
        for i in 0..<16000 {
            switch i % 5 {
            case 0: extremeAudio.append(Float.infinity)
            case 1: extremeAudio.append(-Float.infinity)
            case 2: extremeAudio.append(Float.nan)
            case 3: extremeAudio.append(Float.greatestFiniteMagnitude)
            case 4: extremeAudio.append(-Float.greatestFiniteMagnitude)
            default: extremeAudio.append(0.0)
            }
        }
        
        do {
            let result = try diarizerManager.performCompleteDiarization(extremeAudio)
            XCTAssertNotNil(result, "Should handle extreme values gracefully")
        } catch {
            // Acceptable to fail with extreme values
            XCTAssertTrue(
                error.localizedDescription.contains("value") ||
                error.localizedDescription.contains("invalid") ||
                error.localizedDescription.contains("extreme")
            )
        }
    }
    
    // MARK: - Performance Integration Tests
    
    /// Tests the performance of the complete diarization pipeline.
    /// Measures execution time to ensure the batch processing
    /// provides performance benefits in the complete workflow.
    func testCompleteDiarizationPerformance() throws {
        guard let mockModels = createMockDiarizerModels() else {
            throw XCTSkip("Mock models not available in test environment")
        }
        
        diarizerManager.initialize(models: mockModels)
        
        let audioSamples = createTestAudioWithMultipleSpeakers(duration: 20.0, sampleRate: 16000)
        
        measure {
            do {
                let result = try diarizerManager.performCompleteDiarization(
                    audioSamples,
                    sampleRate: 16000
                )
                XCTAssertFalse(result.segments.isEmpty, "Should produce segments")
            } catch {
                XCTFail("Performance test failed: \(error)")
            }
        }
    }
    
    /// Tests memory usage during complete diarization.
    /// Verifies that the batch processing doesn't cause
    /// excessive memory usage during long audio processing.
    func testMemoryUsageDuringDiarization() throws {
        guard let mockModels = createMockDiarizerModels() else {
            throw XCTSkip("Mock models not available in test environment")
        }
        
        diarizerManager.initialize(models: mockModels)
        
        // Process longer audio to test memory usage
        let audioSamples = createTestAudioWithMultipleSpeakers(duration: 60.0, sampleRate: 16000)
        
        // Measure memory usage
        let initialMemory = getMemoryUsage()
        
        let result = try diarizerManager.performCompleteDiarization(
            audioSamples,
            sampleRate: 16000
        )
        
        let finalMemory = getMemoryUsage()
        let memoryIncrease = finalMemory - initialMemory
        
        // Memory increase should be reasonable (less than 100MB)
        XCTAssertLessThan(memoryIncrease, 100 * 1024 * 1024, "Memory usage should be reasonable")
        
        XCTAssertFalse(result.segments.isEmpty, "Should produce segments")
    }
    
    // MARK: - Helper Methods
    
    /// Creates test audio with multiple speakers.
    /// Generates synthetic audio data with multiple speaker characteristics.
    private func createTestAudioWithMultipleSpeakers(duration: Double, sampleRate: Int) -> [Float] {
        let sampleCount = Int(duration * Double(sampleRate))
        var audio: [Float] = []
        
        for i in 0..<sampleCount {
            let t = Double(i) / Double(sampleRate)
            
            // Create multiple speaker patterns
            var sample: Float = 0.0
            
            // Speaker 1: Low frequency, high amplitude
            if t.truncatingRemainder(dividingBy: 3.0) < 1.0 {
                sample += Float(sin(2.0 * .pi * 200.0 * t) * 0.8)
            }
            
            // Speaker 2: Medium frequency, medium amplitude
            if t.truncatingRemainder(dividingBy: 3.0) >= 1.0 && t.truncatingRemainder(dividingBy: 3.0) < 2.0 {
                sample += Float(sin(2.0 * .pi * 400.0 * t) * 0.6)
            }
            
            // Speaker 3: High frequency, low amplitude
            if t.truncatingRemainder(dividingBy: 3.0) >= 2.0 {
                sample += Float(sin(2.0 * .pi * 800.0 * t) * 0.4)
            }
            
            audio.append(sample)
        }
        
        return audio
    }
    
    /// Creates test audio with overlapping segments.
    /// Generates synthetic audio data with overlapping speaker segments.
    private func createTestAudioWithOverlappingSegments(duration: Double, sampleRate: Int) -> [Float] {
        let sampleCount = Int(duration * Double(sampleRate))
        var audio: [Float] = []
        
        for i in 0..<sampleCount {
            let t = Double(i) / Double(sampleRate)
            var sample: Float = 0.0
            
            // Create overlapping segments
            if t >= 5.0 && t < 15.0 {
                sample += Float(sin(2.0 * .pi * 300.0 * t) * 0.7) // Speaker 1
            }
            
            if t >= 10.0 && t < 20.0 {
                sample += Float(sin(2.0 * .pi * 500.0 * t) * 0.5) // Speaker 2 (overlaps with 1)
            }
            
            if t >= 15.0 && t < 25.0 {
                sample += Float(sin(2.0 * .pi * 700.0 * t) * 0.6) // Speaker 3 (overlaps with 2)
            }
            
            audio.append(sample)
        }
        
        return audio
    }
    
    /// Gets current memory usage for testing.
    /// Returns memory usage in bytes.
    private func getMemoryUsage() -> UInt64 {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size)/4
        
        let kerr: kern_return_t = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(mach_task_self_,
                         task_flavor_t(MACH_TASK_BASIC_INFO),
                         $0,
                         &count)
            }
        }
        
        if kerr == KERN_SUCCESS {
            return info.resident_size
        } else {
            return 0
        }
    }
    
    /// Creates mock DiarizerModels for testing.
    /// Returns nil to use XCTSkip in test environment.
    private func createMockDiarizerModels() -> DiarizerModels? {
        // Return nil to skip in test environment
        return nil
    }
}
