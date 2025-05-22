import struct

import numpy as np


class ReadWriteUp():
    header_size = 16

    def read_up_file(self, file_path: str,
                     dtype: type[np.uint8] | type[np.uint16]) -> np.ndarray:
        """Reads a file containing up1 or up2 patterns.

        Args:
            file_path (str): Path to the file.
            dtype (type[np.uint8] | type[np.uint16]): np.uint8 for up1: N=8 or
                np.uint16 for up2.

        Returns:
            np.ndarray: Extracted patterns [N, 1, m, n].
        """
        assert dtype in [np.uint8, np.uint16], \
            "dtype must be np.uint8 or np.uint16"
        with open(file_path, 'rb') as up_file:
            header = struct.unpack('4i', up_file.read(self.header_size))
            width = header[1]  # Width of patterns in pixels
            height = header[2]  # Height of patterns in pixels
            offset = header[3]  # Offset to first pattern
            pats = np.fromfile(
                up_file,
                dtype=dtype,
                offset=offset-self.header_size)
            num_pats = int(pats.shape[0] / (width * height))
        return pats.reshape(num_pats, 1, height, width)

    def write_up_file(
            self,
            pat_size: int,
            file_path: str,
            write_pats: np.ndarray,
            dtype: type[np.uint8] | type[np.uint16]):
        """Writes a file containing up1 or up2 patterns.

        Args:
            pat_size (int): Size of patterns [N, 1, m, n].
            file_path (str): Path and name for new file.
            write_pats (np.ndarray): Patterns to write.
            dtype (type[np.uint8] | type[np.uint16]): Desired type of data in
                'uint8' or 'uint16' format.
        """
        assert dtype in [np.uint8, np.uint16], \
            "dtype must be np.uint8 or np.uint16"
        if write_pats.dtype == dtype:
            with open(file_path, 'wb') as up_file:
                header = struct.pack(
                    '4i', 1, pat_size, pat_size, self.header_size)
                up_file.write(header)
                write_pats.tofile(up_file)
        else:
            print(
                'ERROR: Type of patterns ({}) does not match \
                with desired type ({}))'.format(write_pats.dtype, dtype))
