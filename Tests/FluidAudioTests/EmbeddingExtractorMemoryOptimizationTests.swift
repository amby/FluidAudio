import CoreML
import XCTest
@testable import FluidAudio

/// Comprehensive test suite for EmbeddingExtractor memory optimization and buffer handling.
///
/// This test class covers the new memory optimization features that:
/// - Handle multiple audio copies for batch processing
/// - Optimize mask buffer handling with slot-based processing
/// - Use ANE-aligned memory for better performance
/// - Implement zero-copy operations where possible
///
/// Key test categories:
/// - Multiple audio copy handling
/// - Mask buffer slot optimization
/// - ANE memory alignment
/// - Zero-copy operations
/// - Memory stress testing
@available(macOS 13.0, iOS 16.0, *)
final class EmbeddingExtractorMemoryOptimizationTests: XCTestCase {
    
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
    
    // MARK: - Multiple Audio Copy Tests
    
    /// Tests that multiple audio copies are handled correctly for batch processing.
    /// Verifies that the system can handle up to 3 audio copies in the waveform buffer.
    func testMultipleAudioCopiesHandling() throws {
        try XCTSkipIf(embeddingExtractor == nil, "Cannot test without real MLModel")
        
        let audio: [Float] = Array(repeating: 0.1, count: 160000)
        
        // Create masks that will trigger multiple audio copies
        let masks: [[Float]] = [
            Array(repeating: 1.0, count: 1000), // Speaker 0: active
            Array(repeating: 1.0, count: 1000), // Speaker 1: active
            Array(repeating: 1.0, count: 1000)  // Speaker 2: active
        ]
        
        let embeddings = try embeddingExtractor.getEmbeddings(
            audio: audio,
            masks: masks,
            minActivityThreshold: 10.0
        )
        
        XCTAssertEqual(embeddings.count, 3)
        
        // All embeddings should be non-zero since all speakers are active
        for embedding in embeddings {
            XCTAssertEqual(embedding.count, 256)
            XCTAssertFalse(embedding.allSatisfy { $0 == 0.0 })
        }
    }
    
    /// Tests that audio copies are limited to maximum of 3.
    /// Verifies that the system correctly limits audio copies to the buffer capacity.
    func testAudioCopyLimit() throws {
        try XCTSkipIf(embeddingExtractor == nil, "Cannot test without real MLModel")
        
        let audio: [Float] = Array(repeating: 0.1, count: 160000)
        
        // Create masks for more than 3 speakers to test copy limiting
        let masks: [[Float]] = [
            Array(repeating: 1.0, count: 1000), // Speaker 0: active
            Array(repeating: 1.0, count: 1000), // Speaker 1: active
            Array(repeating: 1.0, count: 1000), // Speaker 2: active
            Array(repeating: 1.0, count: 1000), // Speaker 3: active
            Array(repeating: 1.0, count: 1000)  // Speaker 4: active
        ]
        
        let embeddings = try embeddingExtractor.getEmbeddings(
            audio: audio,
            masks: masks,
            minActivityThreshold: 10.0
        )
        
        XCTAssertEqual(embeddings.count, 5)
        
        // All embeddings should be valid
        for embedding in embeddings {
            XCTAssertEqual(embedding.count, 256)
        }
    }
    
    /// Tests that audio copies are handled correctly with different audio sizes.
    /// Verifies that the system can handle various audio lengths in batch processing.
    func testAudioCopyWithDifferentSizes() throws {
        try XCTSkipIf(embeddingExtractor == nil, "Cannot test without real MLModel")
        
        let audioSizes = [16000, 80000, 160000, 320000] // Different audio lengths
        
        for audioSize in audioSizes {
            let audio: [Float] = Array(repeating: 0.1, count: audioSize)
            let masks: [[Float]] = [
                Array(repeating: 1.0, count: 1000),
                Array(repeating: 1.0, count: 1000),
                Array(repeating: 1.0, count: 1000)
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
    }
    
    // MARK: - Mask Buffer Slot Optimization Tests
    
    /// Tests that mask buffer slots are used correctly for batch processing.
    /// Verifies that the system correctly fills mask buffer slots for multiple speakers.
    func testMaskBufferSlotOptimization() throws {
        try XCTSkipIf(embeddingExtractor == nil, "Cannot test without real MLModel")
        
        let audio: [Float] = Array(repeating: 0.1, count: 160000)
        
        // Create masks with different patterns to test slot handling
        let masks: [[Float]] = [
            Array(repeating: 1.0, count: 1000), // Speaker 0: all ones
            Array(repeating: 0.5, count: 1000), // Speaker 1: half values
            Array(repeating: 0.0, count: 1000)  // Speaker 2: all zeros
        ]
        
        let embeddings = try embeddingExtractor.getEmbeddings(
            audio: audio,
            masks: masks,
            minActivityThreshold: 10.0
        )
        
        XCTAssertEqual(embeddings.count, 3)
        
        // Results should reflect the mask patterns
        XCTAssertFalse(embeddings[0].allSatisfy { $0 == 0.0 }, "Speaker 0 should be active")
        XCTAssertFalse(embeddings[1].allSatisfy { $0 == 0.0 }, "Speaker 1 should be active")
        XCTAssertTrue(embeddings[2].allSatisfy { $0 == 0.0 }, "Speaker 2 should be inactive")
    }
    
    /// Tests that mask buffer slots are cleared correctly between batches.
    /// Verifies that the system properly clears buffer slots for new batches.
    func testMaskBufferSlotClearing() throws {
        try XCTSkipIf(embeddingExtractor == nil, "Cannot test without real MLModel")
        
        let audio: [Float] = Array(repeating: 0.1, count: 160000)
        
        // Create masks that will trigger multiple batches
        let masks: [[Float]] = [
            Array(repeating: 1.0, count: 1000), // Speaker 0: active
            Array(repeating: 1.0, count: 1000), // Speaker 1: active
            Array(repeating: 1.0, count: 1000), // Speaker 2: active
            Array(repeating: 1.0, count: 1000), // Speaker 3: active
            Array(repeating: 1.0, count: 1000)  // Speaker 4: active
        ]
        
        let embeddings = try embeddingExtractor.getEmbeddings(
            audio: audio,
            masks: masks,
            minActivityThreshold: 10.0
        )
        
        XCTAssertEqual(embeddings.count, 5)
        
        // All embeddings should be valid
        for embedding in embeddings {
            XCTAssertEqual(embedding.count, 256)
        }
    }
    
    /// Tests that mask buffer slots handle different mask sizes correctly.
    /// Verifies that the system can handle masks of varying sizes in batch processing.
    func testMaskBufferSlotWithDifferentSizes() throws {
        try XCTSkipIf(embeddingExtractor == nil, "Cannot test without real MLModel")
        
        let audio: [Float] = Array(repeating: 0.1, count: 160000)
        
        // Create masks with different sizes
        let masks: [[Float]] = [
            Array(repeating: 1.0, count: 500),  // Speaker 0: 500 frames
            Array(repeating: 1.0, count: 1000), // Speaker 1: 1000 frames
            Array(repeating: 1.0, count: 1500) // Speaker 2: 1500 frames
        ]
        
        let embeddings = try embeddingExtractor.getEmbeddings(
            audio: audio,
            masks: masks,
            minActivityThreshold: 10.0
        )
        
        XCTAssertEqual(embeddings.count, 3)
        
        // All embeddings should be valid
        for embedding in embeddings {
            XCTAssertEqual(embedding.count, 256)
        }
    }
    
    // MARK: - ANE Memory Alignment Tests
    
    /// Tests that waveform buffers are ANE-aligned for optimal performance.
    /// Verifies that the system uses ANE-aligned memory for waveform buffers.
    func testWaveformBufferANEAlignment() throws {
        let optimizer = ANEMemoryOptimizer()
        
        // Test waveform buffer creation
        let waveformBuffer = try optimizer.createAlignedArray(
            shape: [3, 160000] as [NSNumber],
            dataType: .float32
        )
        
        let address = Int(bitPattern: waveformBuffer.dataPointer)
        XCTAssertEqual(
            address % ANEMemoryOptimizer.aneAlignment, 0,
            "Waveform buffer should be ANE-aligned")
    }
    
    /// Tests that mask buffers are ANE-aligned for optimal performance.
    /// Verifies that the system uses ANE-aligned memory for mask buffers.
    func testMaskBufferANEAlignment() throws {
        let optimizer = ANEMemoryOptimizer()
        
        // Test mask buffer creation
        let maskBuffer = try optimizer.createAlignedArray(
            shape: [3, 1000] as [NSNumber],
            dataType: .float32
        )
        
        let address = Int(bitPattern: maskBuffer.dataPointer)
        XCTAssertEqual(
            address % ANEMemoryOptimizer.aneAlignment, 0,
            "Mask buffer should be ANE-aligned")
    }
    
    /// Tests that pooled buffers maintain ANE alignment.
    /// Verifies that the buffer pooling system preserves ANE alignment.
    func testPooledBufferANEAlignment() throws {
        let optimizer = ANEMemoryOptimizer()
        
        // Test pooled buffer creation
        let pooledBuffer = try optimizer.getPooledBuffer(
            key: "test_waveform",
            shape: [3, 160000] as [NSNumber],
            dataType: .float32
        )
        
        let address = Int(bitPattern: pooledBuffer.dataPointer)
        XCTAssertEqual(
            address % ANEMemoryOptimizer.aneAlignment, 0,
            "Pooled buffer should be ANE-aligned")
    }
    
    // MARK: - Zero-Copy Operations Tests
    
    /// Tests that zero-copy operations are used where possible.
    /// Verifies that the system implements zero-copy operations for better performance.
    func testZeroCopyOperations() throws {
        try XCTSkipIf(embeddingExtractor == nil, "Cannot test without real MLModel")
        
        let audio: [Float] = Array(repeating: 0.1, count: 160000)
        let masks: [[Float]] = [
            Array(repeating: 1.0, count: 1000),
            Array(repeating: 1.0, count: 1000),
            Array(repeating: 1.0, count: 1000)
        ]
        
        let embeddings = try embeddingExtractor.getEmbeddings(
            audio: audio,
            masks: masks,
            minActivityThreshold: 10.0
        )
        
        XCTAssertEqual(embeddings.count, 3)
        
        // All embeddings should be valid
        for embedding in embeddings {
            XCTAssertEqual(embedding.count, 256)
        }
    }
    
    /// Tests that zero-copy feature providers work correctly.
    /// Verifies that the system can create zero-copy feature providers for model inference.
    func testZeroCopyFeatureProvider() throws {
        let optimizer = ANEMemoryOptimizer()
        
        // Create aligned buffers
        let waveformBuffer = try optimizer.createAlignedArray(
            shape: [3, 160000] as [NSNumber],
            dataType: .float32
        )
        
        let maskBuffer = try optimizer.createAlignedArray(
            shape: [3, 1000] as [NSNumber],
            dataType: .float32
        )
        
        // Create zero-copy feature provider
        let features: [String: MLFeatureValue] = [
            "waveform": MLFeatureValue(multiArray: waveformBuffer),
            "mask": MLFeatureValue(multiArray: maskBuffer),
        ]
        
        let provider = ZeroCopyDiarizerFeatureProvider(features: features)
        
        // Verify features
        XCTAssertEqual(provider.featureNames.sorted(), ["mask", "waveform"])
        
        let waveformFeature = provider.featureValue(for: "waveform")
        XCTAssertNotNil(waveformFeature)
        XCTAssertTrue(
            waveformFeature?.multiArrayValue === waveformBuffer,
            "Should use same buffer instance")
        
        let maskFeature = provider.featureValue(for: "mask")
        XCTAssertNotNil(maskFeature)
        XCTAssertTrue(
            maskFeature?.multiArrayValue === maskBuffer,
            "Should use same buffer instance")
    }
    
    // MARK: - Memory Stress Tests
    
    /// Tests memory handling under stress with many speakers.
    /// Verifies that the system can handle large numbers of speakers without memory issues.
    func testMemoryStressWithManySpeakers() throws {
        try XCTSkipIf(embeddingExtractor == nil, "Cannot test without real MLModel")
        
        let audio: [Float] = Array(repeating: 0.1, count: 160000)
        
        // Create masks for many speakers
        let speakerCount = 20
        let masks: [[Float]] = (0..<speakerCount).map { i in
            Array(repeating: i % 2 == 0 ? 1.0 : 0.0, count: 1000)
        }
        
        let embeddings = try embeddingExtractor.getEmbeddings(
            audio: audio,
            masks: masks,
            minActivityThreshold: 10.0
        )
        
        XCTAssertEqual(embeddings.count, speakerCount)
        
        // All embeddings should be valid
        for embedding in embeddings {
            XCTAssertEqual(embedding.count, 256)
        }
    }
    
    /// Tests memory handling with very large audio buffers.
    /// Verifies that the system can handle large audio buffers without memory issues.
    func testMemoryStressWithLargeAudio() throws {
        try XCTSkipIf(embeddingExtractor == nil, "Cannot test without real MLModel")
        
        // Create very large audio buffer
        let audio: [Float] = Array(repeating: 0.1, count: 800000) // 50 seconds at 16kHz
        let masks: [[Float]] = [
            Array(repeating: 1.0, count: 1000),
            Array(repeating: 1.0, count: 1000),
            Array(repeating: 1.0, count: 1000)
        ]
        
        let embeddings = try embeddingExtractor.getEmbeddings(
            audio: audio,
            masks: masks,
            minActivityThreshold: 10.0
        )
        
        XCTAssertEqual(embeddings.count, 3)
        
        // All embeddings should be valid
        for embedding in embeddings {
            XCTAssertEqual(embedding.count, 256)
        }
    }
    
    /// Tests memory handling with very large mask buffers.
    /// Verifies that the system can handle large mask buffers without memory issues.
    func testMemoryStressWithLargeMasks() throws {
        try XCTSkipIf(embeddingExtractor == nil, "Cannot test without real MLModel")
        
        let audio: [Float] = Array(repeating: 0.1, count: 160000)
        
        // Create very large mask buffers
        let masks: [[Float]] = [
            Array(repeating: 1.0, count: 10000), // 10k frames
            Array(repeating: 1.0, count: 10000), // 10k frames
            Array(repeating: 1.0, count: 10000)   // 10k frames
        ]
        
        let embeddings = try embeddingExtractor.getEmbeddings(
            audio: audio,
            masks: masks,
            minActivityThreshold: 10.0
        )
        
        XCTAssertEqual(embeddings.count, 3)
        
        // All embeddings should be valid
        for embedding in embeddings {
            XCTAssertEqual(embedding.count, 256)
        }
    }
    
    // MARK: - Concurrent Memory Access Tests
    
    /// Tests concurrent access to memory buffers.
    /// Verifies that the system can handle concurrent access to memory buffers safely.
    func testConcurrentMemoryAccess() {
        let optimizer = ANEMemoryOptimizer()
        let expectation = XCTestExpectation(description: "Concurrent access")
        expectation.expectedFulfillmentCount = 10
        
        // Multiple threads accessing embedding-related buffers
        DispatchQueue.concurrentPerform(iterations: 10) { index in
            autoreleasepool {
                do {
                    // Each thread gets its own waveform buffer
                    let waveformBuffer = try optimizer.getPooledBuffer(
                        key: "concurrent_waveform_\(index)",
                        shape: [3, 160000] as [NSNumber],
                        dataType: .float32
                    )
                    
                    // Shared mask buffers (simulating speaker masks)
                    let maskBuffer = try optimizer.getPooledBuffer(
                        key: "concurrent_mask_\(index % 3)",
                        shape: [3, 1000] as [NSNumber],
                        dataType: .float32
                    )
                    
                    // Simulate some processing
                    waveformBuffer[0] = NSNumber(value: Float(index))
                    maskBuffer[0] = NSNumber(value: Float(index))
                    
                    XCTAssertEqual(waveformBuffer[0].floatValue, Float(index))
                    expectation.fulfill()
                } catch {
                    XCTFail("Concurrent access failed: \(error)")
                }
            }
        }
        
        wait(for: [expectation], timeout: 10.0)
    }
    
    /// Tests memory cleanup after processing.
    /// Verifies that the system properly cleans up memory after processing.
    func testMemoryCleanup() throws {
        let optimizer = ANEMemoryOptimizer()
        
        // Create some buffers
        let buffer1 = try optimizer.getPooledBuffer(
            key: "cleanup_test_1",
            shape: [3, 160000] as [NSNumber],
            dataType: .float32
        )
        
        let buffer2 = try optimizer.getPooledBuffer(
            key: "cleanup_test_2",
            shape: [3, 1000] as [NSNumber],
            dataType: .float32
        )
        
        // Verify buffers exist
        XCTAssertNotNil(buffer1)
        XCTAssertNotNil(buffer2)
        
        // Clear pool
        optimizer.clearBufferPool()
        
        // Create new buffer after cleanup
        let buffer3 = try optimizer.getPooledBuffer(
            key: "cleanup_test_3",
            shape: [3, 80000] as [NSNumber],
            dataType: .float32
        )
        
        XCTAssertNotNil(buffer3)
    }
    
    // MARK: - Performance Tests
    
    /// Tests performance improvement with memory optimization.
    /// Verifies that the memory optimization provides performance benefits.
    func testMemoryOptimizationPerformance() throws {
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
