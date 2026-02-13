import collections
import statistics
import math

class PeakDetector:
    """
    Real-time peak detection for streaming data using Z-Score and Moving Average.
    
    This class analyzes a stream of float values (e.g., similarity scores) to identify
    statistically significant peaks or values exceeding an absolute threshold.
    It uses a sliding window to adapt to the data distribution.
    """

    def __init__(self, window_size: int = 30, z_threshold: float = 3.5, absolute_threshold: float = 0.95):
        """
        Initialize the PeakDetector.

        Args:
            window_size (int): The number of recent data points to keep for statistical analysis.
            z_threshold (float): The number of standard deviations above the mean to consider a peak.
            absolute_threshold (float): An absolute value that, if exceeded, always triggers a peak detection.
        """
        self.window_size = window_size
        self.z_threshold = z_threshold
        self.absolute_threshold = absolute_threshold
        self.window = collections.deque(maxlen=window_size)
    
    def add(self, value: float) -> dict:
        """
        Process a new value from the stream and determine if it represents a peak.

        Args:
            value (float): The next value in the stream (expected to be a similarity score).

        Returns:
            dict: A dictionary containing:
                - "is_peak" (bool): True if the value is considered a peak.
                - "reason" (str): The reason for the peak ("absolute_threshold", "z_score", or None).
                - "z_score" (float): The calculated Z-Score (or None if not applicable).
                - "mean" (float): The current moving average (or None).
        """
        result = {
            "is_peak": False,
            "reason": None,
            "z_score": None,
            "mean": None
        }
        
        # 1. Check Absolute Threshold (Emergency/Critical pass)
        if value >= self.absolute_threshold:
            result["is_peak"] = True
            result["reason"] = "absolute_threshold"
            self.window.append(value)
            return result

        # 2. Cold Start / Insufficient Data
        if len(self.window) < 2:
            self.window.append(value)
            return result

        # 3. Statistical Analysis (Z-Score)
        mean = statistics.mean(self.window)
        std_dev = statistics.stdev(self.window) if len(self.window) > 1 else 0.0
        
        result["mean"] = mean

        if std_dev > 0:
            z_score = (value - mean) / std_dev
            result["z_score"] = z_score
            if z_score > self.z_threshold:
                result["is_peak"] = True
                result["reason"] = "z_score"
        
        # Update the window with the new value
        self.window.append(value)
        
        return result

if __name__ == "__main__":
    print("--- Running PeakDetector Tests ---")

    # Test Case 1: Short stream / Cold start
    print("\nTest Case 1: Cold Start & Absolute Threshold")
    detector = PeakDetector(window_size=5, z_threshold=2.0, absolute_threshold=0.90)
    stream_1 = [0.5, 0.6, 0.95, 0.55] 
    for val in stream_1:
        res = detector.add(val)
        print(f"Value: {val:.2f} -> Peak: {res['is_peak']} ({res.get('reason')})")
    
    # Test Case 2: Noisy stream with a significant statistical peak
    print("\nTest Case 2: Statistical Peak in Noise")
    detector = PeakDetector(window_size=10, z_threshold=3.0, absolute_threshold=0.99)
    # Background noise
    noise = [0.1, 0.12, 0.11, 0.09, 0.13, 0.1, 0.11, 0.12, 0.08, 0.1]
    # Sudden peak
    peak_val = 0.5 
    # Back to noise
    post_peak = [0.11, 0.1]
    
    stream_2 = noise + [peak_val] + post_peak
    for val in stream_2:
        res = detector.add(val)
        status = f"PEAK! ({res['reason']}, Z={res['z_score']:.2f})" if res['is_peak'] else ""
        print(f"Value: {val:.2f} -> {res['is_peak']} {status}")
        
    # Test Case 3: Gradual rise
    print("\nTest Case 3: Gradual Rise")
    detector = PeakDetector(window_size=5, z_threshold=3.0)
    stream_3 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    for val in stream_3:
        res = detector.add(val)
        print(f"Value: {val:.2f} -> Peak: {res['is_peak']}")