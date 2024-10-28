import numpy as np

class ATFHandler:
    """
    Handler for Axon Text File (.atf) format used in ion channel recordings.
    Provides functionality to load, parse, and extract data from ATF files.
    """

    def __init__(self, filepath):
        self.filepath = filepath
        self.headers = []
        self.data = None
        self.metadata = {}
        self.time_data = None
        self.current_data = None
        self.sampling_rate = None

    def load_atf(self):
        """
        Load and parse the ATF file.
        Extracts headers, metadata, and numerical data.
        """
        try:
            with open(self.filepath, 'r') as file:
                lines = file.readlines()

                # Verify ATF format
                if not lines[0].startswith('ATF'):
                    raise ValueError("Invalid file format. File must be an ATF file.")

                # Parse header information
                header_info = lines[1].strip().split()
                num_header_lines = int(header_info[0])
                num_columns = int(header_info[1])

                # Extract headers and metadata
                self.headers = []
                metadata_lines = []
                for i in range(2, num_header_lines):
                    line = lines[i].strip()
                    if '"' in line:  # Column headers
                        self.headers = line.replace('"', '').strip().split()
                    else:  # Metadata
                        metadata_lines.append(line)

                # Parse metadata
                self._parse_metadata(metadata_lines)

                # Parse numerical data
                data_lines = lines[num_header_lines:]
                parsed_data = []
                for line in data_lines:
                    try:
                        values = [float(x) for x in line.strip().split()]
                        if len(values) == num_columns:
                            parsed_data.append(values)
                    except ValueError:
                        continue

                self.data = np.array(parsed_data)
                
                # Extract time and current data
                self._extract_data_columns()

                return True

        except Exception as e:
            print(f"Error loading ATF file: {str(e)}")
            return False

    def _parse_metadata(self, metadata_lines):
        """Parse metadata lines into structured format."""
        for line in metadata_lines:
            if '=' in line:
                key, value = line.split('=', 1)
                self.metadata[key.strip()] = value.strip()
            
        # Extract sampling rate if available
        if 'ACQUIRE' in self.metadata:
            acquire_info = self.metadata['ACQUIRE'].split()
            for item in acquire_info:
                if item.startswith('SampleInterval='):
                    interval = float(item.split('=')[1])
                    self.sampling_rate = 1.0 / interval
                    break

    def _extract_data_columns(self):
        """Extract time and current data from the loaded data array."""
        if self.data is None or len(self.data) == 0:
            return

        # Time is typically the first column
        self.time_data = self.data[:, 0]

        # Current is typically the second column
        self.current_data = self.data[:, 1]

    def get_time_data(self):
        """Return the time data array."""
        return self.time_data

    def get_current_data(self):
        """Return the current data array."""
        return self.current_data

    def get_sampling_rate(self):
        """Return the sampling rate in Hz."""
        return self.sampling_rate

    def get_metadata(self):
        """Return all metadata as a dictionary."""
        return self.metadata

    def get_data_info(self):
        """Return basic information about the loaded data."""
        if self.data is None:
            return "No data loaded"

        return {
            "num_points": len(self.data),
            "duration": self.time_data[-1] - self.time_data[0],
            "sampling_rate": self.sampling_rate,
            "num_channels": self.data.shape[1],
            "headers": self.headers
        }

if __name__ == "__main__":
    # Example usage
    handler = ATFHandler("data/202304_0521.atf")
    if handler.load_atf():
        print("Data Info:", handler.get_data_info())
        print("Sampling Rate:", handler.get_sampling_rate(), "Hz")