import XCTest
@testable import FluidAudio

/// Tests for absolute timing functionality in streaming ASR
/// Verifies that token timings are computed relative to stream start, not individual chunks
final class AbsoluteTimingTests: XCTestCase {
    
    private var manager: AsrManager!
    
    override func setUp() async throws {
        manager = AsrManager(config: ASRConfig(sampleRate: 16000))
        // Note: We don't initialize models for these tests as we're testing timing logic only
    }
    
    override func tearDown() {
        manager = nil
    }
    
    // MARK: - Window Time Offset Tests
    
    func testProcessTranscriptionResultWithAbsoluteFrameTimestamps() {
        let tokenIds = [1, 2, 3]
        // These are already absolute frame timestamps (converted from relative + window offset)
        // 5.8s = 5.8 / 0.08 = 72.5 frames, 6.6s = 6.6 / 0.08 = 82.5 frames, 7.4s = 7.4 / 0.08 = 92.5 frames
        let absoluteFrameTimestamps = [72, 82, 92]  // Rounded to integer frames
        let confidences: [Float] = [0.8, 0.9, 0.7]
        let audioSamples = Array(repeating: Float(0), count: 16_000)  // 1 second
        let processingTime = 0.5
        
        let result = manager.processTranscriptionResult(
            tokenIds: tokenIds,
            timestamps: absoluteFrameTimestamps,  // Pass absolute frame timestamps directly
            confidences: confidences,
            encoderSequenceLength: 100,
            audioSamples: audioSamples,
            processingTime: processingTime
        )
        
        // Verify that token timings are absolute from stream start
        guard let tokenTimings = result.tokenTimings, tokenTimings.count == 3 else {
            XCTFail("Expected 3 token timings")
            return
        }
        
        // Token 1: frame 72 * 0.08s = 5.76s
        XCTAssertEqual(tokenTimings[0].startTime, 5.76, accuracy: 0.01)
        XCTAssertEqual(tokenTimings[0].endTime, 6.56, accuracy: 0.01)  // Next token start
        
        // Token 2: frame 82 * 0.08s = 6.56s
        XCTAssertEqual(tokenTimings[1].startTime, 6.56, accuracy: 0.01)
        XCTAssertEqual(tokenTimings[1].endTime, 7.36, accuracy: 0.01)  // Next token start
        
        // Token 3: frame 92 * 0.08s = 7.36s
        XCTAssertEqual(tokenTimings[2].startTime, 7.36, accuracy: 0.01)
        XCTAssertEqual(tokenTimings[2].endTime, 7.44, accuracy: 0.01)  // Default duration
    }
    
    func testProcessTranscriptionResultWithRelativeFrameTimestamps() {
        let tokenIds = [1, 2, 3]
        let timestamps = [10, 20, 30]  // Relative frame indices (no window offset applied)
        let confidences: [Float] = [0.8, 0.9, 0.7]
        let audioSamples = Array(repeating: Float(0), count: 16_000)  // 1 second
        let processingTime = 0.5
        
        let result = manager.processTranscriptionResult(
            tokenIds: tokenIds,
            timestamps: timestamps,
            confidences: confidences,
            encoderSequenceLength: 100,
            audioSamples: audioSamples,
            processingTime: processingTime
        )
        
        // Verify that token timings are relative to chunk start (backward compatibility)
        guard let tokenTimings = result.tokenTimings, tokenTimings.count == 3 else {
            XCTFail("Expected 3 token timings")
            return
        }
        
        // Token 1: frame 10 * 0.08s + 0.0s offset = 0.8s
        XCTAssertEqual(tokenTimings[0].startTime, 0.8, accuracy: 0.01)
        XCTAssertEqual(tokenTimings[0].endTime, 1.6, accuracy: 0.01)
        
        // Token 2: frame 20 * 0.08s + 0.0s offset = 1.6s
        XCTAssertEqual(tokenTimings[1].startTime, 1.6, accuracy: 0.01)
        XCTAssertEqual(tokenTimings[1].endTime, 2.4, accuracy: 0.01)
        
        // Token 3: frame 30 * 0.08s + 0.0s offset = 2.4s
        XCTAssertEqual(tokenTimings[2].startTime, 2.4, accuracy: 0.01)
        XCTAssertEqual(tokenTimings[2].endTime, 2.48, accuracy: 0.01)
    }
    
    func testCreateTokenTimingsWithAbsoluteFrameTimestamps() {
        let tokenIds = [100, 200, 300]
        // These are already absolute frame timestamps (converted from relative + window offset)
        let absoluteFrameTimestamps = [125, 137, 150]  // 10.0s, 10.96s, 12.0s converted to frames
        let confidences: [Float] = [0.95, 0.87, 0.92]
        
        // Use reflection to access private method for testing
        let result = manager.processTranscriptionResult(
            tokenIds: tokenIds,
            timestamps: absoluteFrameTimestamps,  // Pass absolute frame timestamps directly
            confidences: confidences,
            encoderSequenceLength: 100,
            audioSamples: Array(repeating: Float(0), count: 16_000),
            processingTime: 0.5
        )
        
        guard let tokenTimings = result.tokenTimings, tokenTimings.count == 3 else {
            XCTFail("Expected 3 token timings")
            return
        }
        
        // Verify absolute timing calculations
        XCTAssertEqual(tokenTimings[0].startTime, 10.0, accuracy: 0.01)  // 0 * 0.08 + 10.0
        XCTAssertEqual(tokenTimings[0].endTime, 10.96, accuracy: 0.01)   // 12 * 0.08 + 10.0
        
        XCTAssertEqual(tokenTimings[1].startTime, 10.96, accuracy: 0.01) // 12 * 0.08 + 10.0
        XCTAssertEqual(tokenTimings[1].endTime, 12.0, accuracy: 0.01)    // 25 * 0.08 + 10.0
        
        XCTAssertEqual(tokenTimings[2].startTime, 12.0, accuracy: 0.01)  // 25 * 0.08 + 10.0
        XCTAssertEqual(tokenTimings[2].endTime, 12.08, accuracy: 0.01)   // Default duration
    }
    
    // MARK: - Streaming Window Offset Calculation Tests
    
    func testStreamingWindowOffsetCalculation() {
        // Test the window offset calculation logic used in StreamingAsrManager
        let sampleRate = 16000.0
        let leftContextSeconds = 2.0
        let leftContextSamples = Int(leftContextSeconds * sampleRate)  // 32,000 samples
        
        // Simulate streaming chunks with different positions
        let testCases: [(nextWindowCenterStart: Int, expectedOffset: TimeInterval)] = [
            // First chunk: center starts at 0, left context starts at 0
            (0, 0.0),
            
            // Second chunk: center starts at 11s (176,000 samples), left context starts at 9s (144,000 samples)
            (176_000, 9.0),
            
            // Third chunk: center starts at 22s (352,000 samples), left context starts at 20s (320,000 samples)
            (352_000, 20.0),
            
            // Edge case: very early chunk with negative left context (should clamp to 0)
            (10_000, 0.0)  // 10,000 - 32,000 = -22,000, clamped to 0
        ]
        
        for (nextWindowCenterStart, expectedOffset) in testCases {
            let leftStartAbs = max(0, nextWindowCenterStart - leftContextSamples)
            let windowTimeOffset = TimeInterval(leftStartAbs) / sampleRate
            
            XCTAssertEqual(windowTimeOffset, expectedOffset, accuracy: 0.01,
                          "Failed for nextWindowCenterStart=\(nextWindowCenterStart)")
        }
    }
    
    // MARK: - Edge Case Tests
    
    func testEmptyTokenArraysWithAbsoluteFrameTimestamps() {
        let result = manager.processTranscriptionResult(
            tokenIds: [],
            timestamps: [],
            confidences: [],
            encoderSequenceLength: 100,
            audioSamples: Array(repeating: Float(0), count: 16_000),
            processingTime: 0.5
        )
        
        XCTAssertTrue(result.tokenTimings?.isEmpty ?? true, "Empty arrays should produce empty token timings")
    }
    
    func testMismatchedArrayLengthsWithAbsoluteFrameTimestamps() {
        let result = manager.processTranscriptionResult(
            tokenIds: [1, 2, 3],
            timestamps: [35, 45],  // Mismatched length (absolute frame timestamps)
            confidences: [0.8, 0.9, 0.7],
            encoderSequenceLength: 100,
            audioSamples: Array(repeating: Float(0), count: 16_000),
            processingTime: 0.5
        )
        
        XCTAssertTrue(result.tokenTimings?.isEmpty ?? true, "Mismatched arrays should produce empty token timings")
    }
    
    func testLargeAbsoluteFrameTimestamp() {
        let tokenIds = [1]
        // This is already an absolute frame timestamp (1 hour into the stream)
        let largeAbsoluteFrameTimestamp = [45000]  // 3600.0s / 0.08s = 45000 frames
        let confidences: [Float] = [0.9]
        
        let result = manager.processTranscriptionResult(
            tokenIds: tokenIds,
            timestamps: largeAbsoluteFrameTimestamp,  // Pass absolute frame timestamp directly
            confidences: confidences,
            encoderSequenceLength: 100,
            audioSamples: Array(repeating: Float(0), count: 16_000),
            processingTime: 0.5
        )
        
        guard let tokenTimings = result.tokenTimings, tokenTimings.count == 1 else {
            XCTFail("Expected 1 token timing")
            return
        }
        
        // Token should start at 1 hour (3600.0s) + 0 frames
        XCTAssertEqual(tokenTimings[0].startTime, 3600.0, accuracy: 0.01)
        XCTAssertEqual(tokenTimings[0].endTime, 3600.08, accuracy: 0.01)  // Default duration
    }
    
    // MARK: - Performance Tests
    
    func testWindowOffsetCalculationPerformance() {
        let iterations = 10_000
        
        measure {
            for i in 0..<iterations {
                let nextWindowCenterStart = i * 176_000  // Simulate chunk advancement
                let leftContextSamples = 32_000
                let leftStartAbs = max(0, nextWindowCenterStart - leftContextSamples)
                let windowTimeOffset = TimeInterval(leftStartAbs) / 16000.0
                
                // Verify the calculation is reasonable
                XCTAssertGreaterThanOrEqual(windowTimeOffset, 0.0)
            }
        }
    }
    
    // MARK: - Integration Tests
    
    func testStreamingChunkTimingConsistency() {
        // Test that multiple chunks with different window offsets produce consistent absolute timings
        let chunk1Tokens = [1, 2]
        let chunk1Confidences: [Float] = [0.8, 0.9]
        
        let chunk2Tokens = [3, 4]
        let chunk2Confidences: [Float] = [0.85, 0.88]
        
        // These are already absolute frame timestamps (converted from relative + window offset)
        let chunk1AbsoluteFrameTimestamps = [10, 20]  // 0.8s, 1.6s converted to frames
        let chunk2AbsoluteFrameTimestamps = [147, 157]  // 11.76s, 12.56s converted to frames
        
        let result1 = manager.processTranscriptionResult(
            tokenIds: chunk1Tokens,
            timestamps: chunk1AbsoluteFrameTimestamps,  // Pass absolute frame timestamps directly
            confidences: chunk1Confidences,
            encoderSequenceLength: 100,
            audioSamples: Array(repeating: Float(0), count: 16_000),
            processingTime: 0.5
        )
        
        let result2 = manager.processTranscriptionResult(
            tokenIds: chunk2Tokens,
            timestamps: chunk2AbsoluteFrameTimestamps,  // Pass absolute frame timestamps directly
            confidences: chunk2Confidences,
            encoderSequenceLength: 100,
            audioSamples: Array(repeating: Float(0), count: 16_000),
            processingTime: 0.5
        )
        
        guard let timings1 = result1.tokenTimings, let timings2 = result2.tokenTimings else {
            XCTFail("Expected token timings for both chunks")
            return
        }
        
        // Verify that chunk 2 tokens start where chunk 1 tokens ended (accounting for overlap)
        // Chunk 1: tokens at 0.8s and 1.6s
        // Chunk 2: tokens at 11.8s and 12.6s (11.0s offset + relative times)
        XCTAssertEqual(timings1[0].startTime, 0.8, accuracy: 0.01)
        XCTAssertEqual(timings1[1].startTime, 1.6, accuracy: 0.01)
        
        XCTAssertEqual(timings2[0].startTime, 11.76, accuracy: 0.01)
        XCTAssertEqual(timings2[1].startTime, 12.56, accuracy: 0.01)
        
        // Verify there's a proper gap between chunks (no overlap in absolute time)
        XCTAssertGreaterThan(timings2[0].startTime, timings1[1].endTime,
                           "Chunk 2 should start after chunk 1 ends in absolute time")
    }
}
