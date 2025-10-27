"""
Custom video compression implementation using DCT, quantization, and Huffman coding.
"""

import torch
import torch.nn.functional as F
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image
import numpy as np
import os
import shutil
import unittest
from collections import Counter
import io
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

try:
    from torch_dct import dct_2d, idct_2d
except ImportError:
    print("Warning: torch_dct not available. Custom compression functions will not work.")
    dct_2d = None
    idct_2d = None

try:
    import huffman
except ImportError:
    print("Warning: huffman package not available. Custom compression functions will not work.")
    huffman = None


class DCTQuantizer:
    """
    Compresses residuals using a DCT, quantization, and Huffman coding pipeline.
    """
    def __init__(self, block_size=16, quality=50):
        self.block_size = block_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Standard JPEG quantization matrix, scaled for our block size and quality
        q_matrix_8 = torch.tensor([
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99]
        ], dtype=torch.float32)

        # Scale the 8x8 matrix to the target block size
        if block_size % 8 != 0:
            raise ValueError("Block size must be a multiple of 8")
        self.quantization_matrix = q_matrix_8.repeat_interleave(block_size // 8, dim=0).repeat_interleave(block_size // 8, dim=1)
        
        # Adjust for quality
        scale = (5000 / quality) if quality < 50 else (200 - 2 * quality)
        # For highest quality reconstruction, scale should be 0:
        # - When quality = 100: scale = 200 - 2*100 = 0
        # - Lower scale values mean less quantization (division by smaller numbers)
        # - scale = 0 would mean no quantization, but clamp(min=1) prevents this
        # - So practically, scale approaches 0 as quality approaches 100
        self.quantization_matrix = (self.quantization_matrix * scale / 100).clamp(min=1).to(self.device)

    def compress(self, residual_frame):
        """
        Compress a single residual frame.
        """
        if dct_2d is None or huffman is None:
            raise ImportError("Required packages (torch_dct, huffman) not available")
            
        c, h, w = residual_frame.shape
        num_blocks_h = h // self.block_size
        num_blocks_w = w // self.block_size
        
        # --- 1. DCT ---
        # Reshape into blocks and apply DCT
        residual_blocks = residual_frame.reshape(c, num_blocks_h, self.block_size, num_blocks_w, self.block_size)
        residual_blocks = residual_blocks.permute(1, 3, 0, 2, 4).reshape(-1, c, self.block_size, self.block_size)
        dct_coeffs = dct_2d(residual_blocks.to(self.device))
        
        # --- 2. Quantization ---
        quantized_coeffs = torch.round(dct_coeffs / self.quantization_matrix)
        
        # --- 3. Entropy Coding ---
        # Convert to a list of integers for Huffman coding
        quantized_list = quantized_coeffs.cpu().numpy().flatten().astype(np.int16).tolist()
        
        # The huffman library expects a list of (symbol, weight) pairs.
        symbol_weights = Counter(quantized_list).items()
        
        # Handle the edge case of a blank residual (only one symbol)
        if len(symbol_weights) < 2:
            symbol_weights = list(symbol_weights) + [('dummy_symbol', 1)]
        
        huffman_codebook = huffman.codebook(symbol_weights)
        huffman_codebook.pop('dummy_symbol', None)

        encoded_string = "".join(huffman_codebook[val] for val in quantized_list)
        
        return {
            'huffman_codebook': huffman_codebook,
            'encoded_string': encoded_string,
            'original_shape': (c, h, w)
        }

    def decompress(self, compressed_data):
        """
        Decompress a single residual frame from its compressed data.
        """
        if dct_2d is None or huffman is None:
            raise ImportError("Required packages (torch_dct, huffman) not available")
            
        huffman_codebook = compressed_data['huffman_codebook']
        encoded_string = compressed_data['encoded_string']
        c, h, w = compressed_data['original_shape']
        
        # Handle case where the residual was blank
        if not huffman_codebook:
             return torch.zeros((c, h, w))

        # --- 1. Huffman Decoding ---
        inverted_codebook = {v: k for k, v in huffman_codebook.items()}
        decoded_list = []
        current_code = ""
        for bit in encoded_string:
            current_code += bit
            if current_code in inverted_codebook:
                decoded_list.append(inverted_codebook[current_code])
                current_code = ""

        quantized_coeffs = torch.tensor(decoded_list, dtype=torch.float32).reshape(-1, c, self.block_size, self.block_size).to(self.device)
        
        # --- 2. De-quantization ---
        dequantized_coeffs = quantized_coeffs * self.quantization_matrix
        
        # --- 3. Inverse DCT ---
        reconstructed_blocks = idct_2d(dequantized_coeffs)
        
        # Reassemble blocks into a frame
        num_blocks_h = h // self.block_size
        num_blocks_w = w // self.block_size
        reconstructed_residual = reconstructed_blocks.reshape(num_blocks_h, num_blocks_w, c, self.block_size, self.block_size)
        reconstructed_residual = reconstructed_residual.permute(2, 0, 3, 1, 4).reshape(c, h, w)
        
        return reconstructed_residual.cpu()


class VideoCompressor:
    def __init__(self, block_size=16, motion_search_range=16, quality=50):
        self.block_size = block_size
        self.motion_search_range = motion_search_range
        self.quality = quality
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        self.quantizer = DCTQuantizer(block_size=self.block_size, quality=self.quality)
        self.to_tensor = ToTensor()
        self.to_pil = ToPILImage()

    def _get_residuals(self, f_i, f_j, motion_vectors):
        """
        Calculate the residuals between two frames given motion vectors.
        """
        k = self.block_size
        frame_h, frame_w = f_i.shape[1], f_i.shape[2]
        # Derive loop bounds from the motion_vectors tensor to prevent crashes
        num_blocks_h, num_blocks_w, _ = motion_vectors.shape
        
        predicted_f_j = torch.zeros_like(f_j)
        
        for i in range(num_blocks_h):
            for j in range(num_blocks_w):
                mv = motion_vectors[i, j]
                y_start, x_start = i * k, j * k
                
                ref_y_start = y_start + mv[0].item()
                ref_x_start = x_start + mv[1].item()
                
                # Boundary check for the reference block
                ref_y_start_c = np.clip(ref_y_start, 0, frame_h - k)
                ref_x_start_c = np.clip(ref_x_start, 0, frame_w - k)
                
                # Check if the block is within the predicted frame bounds
                if y_start + k <= frame_h and x_start + k <= frame_w:
                    predicted_f_j[:, y_start:y_start+k, x_start:x_start+k] = f_i[:, ref_y_start_c:ref_y_start_c+k, ref_x_start_c:ref_x_start_c+k]

        return f_j - predicted_f_j

    def _calculate_motion_vectors(self, f_i, f_j):
        """
        Calculate motion vector indices from f_i to f_j.
        """
        k = self.motion_search_range
        bk = self.block_size
        
        f_i, f_j = f_i.to(self.device), f_j.to(self.device)
        
        c, h, w = f_i.shape
        num_blocks_h = h // bk
        num_blocks_w = w // bk

        shifts = []
        for dy in [-k, 0, k]:
            for dx in [-k, 0, k]:
                shifted_f_i = torch.roll(f_i, shifts=(dy, dx), dims=(1, 2))
                shifts.append(shifted_f_i)
        
        shifts = torch.stack(shifts)

        shifts_blocked = shifts.unfold(2, bk, bk).unfold(3, bk, bk).permute(0, 2, 4, 1, 3, 5)
        f_j_blocked = f_j.unfold(1, bk, bk).unfold(2, bk, bk).permute(1, 3, 0, 2, 4)

        mse = torch.mean((shifts_blocked - f_j_blocked.unsqueeze(0))**2, dim=(-1, -2, -3))

        best_shift_indices = torch.argmin(mse, dim=0)

        return best_shift_indices.cpu()

    def compress(self, video_tensor, output_path):
        """
        Compresses a video tensor into a custom binary format.
        """
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        os.makedirs(output_path)
        
        iframe_tensor = video_tensor[0]
        iframe_img = self.to_pil(iframe_tensor)
        iframe_img.save(os.path.join(output_path, "0000.jpg"), quality=95)
        
        # We need to read it back to simulate the exact lossy reconstruction
        with open(os.path.join(output_path, "0000.jpg"), 'rb') as f:
            reconstructed_frame = self.to_tensor(Image.open(f).convert("RGB"))
        
        # Reconstruct motion vectors from indices
        shift_vectors = []
        k_range = self.motion_search_range
        for dy in [-k_range, 0, k_range]:
            for dx in [-k_range, 0, k_range]:
                shift_vectors.append(torch.tensor([dy, dx]))
        shift_vectors = torch.stack(shift_vectors)

        for i in tqdm(range(1, video_tensor.shape[0])):
            current_frame = video_tensor[i]
            motion_indices = self._calculate_motion_vectors(reconstructed_frame, current_frame)
            
            motion_vectors = shift_vectors[motion_indices.reshape(-1)].reshape(motion_indices.shape[0], motion_indices.shape[1], 2)
            residuals = self._get_residuals(reconstructed_frame.to(self.device), current_frame.to(self.device), motion_vectors.to(self.device)).cpu()
            
            compressed_residuals = self.quantizer.compress(residuals)
            
            # --- Write to a compact binary file ---
            with open(os.path.join(output_path, f"{i:04d}.bin"), 'wb') as f:
                h_mv, w_mv = motion_indices.shape
                f.write(np.uint16(h_mv).tobytes())
                f.write(np.uint16(w_mv).tobytes())
                f.write(motion_indices.numpy().astype(np.uint8).tobytes())
                
                codebook = compressed_residuals['huffman_codebook']
                encoded_string = compressed_residuals['encoded_string']
                
                f.write(np.uint32(len(codebook)).tobytes())
                for symbol, code in codebook.items():
                    f.write(np.int16(symbol).tobytes())
                    code_bytes = code.encode('ascii')
                    f.write(np.uint8(len(code_bytes)).tobytes())
                    f.write(code_bytes)
                    
                f.write(np.uint64(len(encoded_string)).tobytes())
                padding = '0' * ((8 - len(encoded_string) % 8) % 8)
                bit_string = encoded_string + padding
                byte_array = bytearray(int(bit_string[j:j+8], 2) for j in range(0, len(bit_string), 8))
                f.write(byte_array)

            # Reconstruct frame for next iteration
            decompressed_residuals = self.quantizer.decompress(compressed_residuals)
            predicted_frame_gpu = torch.zeros_like(reconstructed_frame, device=self.device)
            reconstructed_frame_gpu = reconstructed_frame.to(self.device)
            h, w = reconstructed_frame.shape[1], reconstructed_frame.shape[2]
            k = self.block_size
            
            # Use the shape of the motion_vectors tensor for loop bounds to ensure they are synchronized
            num_blocks_h, num_blocks_w, _ = motion_vectors.shape

            for r_idx in range(num_blocks_h):
                for c_idx in range(num_blocks_w):
                    mv = motion_vectors[r_idx, c_idx].to(self.device)
                    y_start, x_start = r_idx * k, c_idx * k
                    ref_y_start, ref_x_start = y_start + mv[0].item(), x_start + mv[1].item()
                    ref_y_start_c = np.clip(ref_y_start, 0, h - k)
                    ref_x_start_c = np.clip(ref_x_start, 0, w - k)
                    if y_start + k <= h and x_start + k <= w:
                        predicted_frame_gpu[:, y_start:y_start+k, x_start:x_start+k] = reconstructed_frame_gpu[:, ref_y_start_c:ref_y_start_c+k, ref_x_start_c:ref_x_start_c+k]
            
            reconstructed_frame = (predicted_frame_gpu + decompressed_residuals.to(self.device)).clamp(0, 1).cpu()

        print("Compression complete.")

    def decompress(self, compressed_path):
        """
        Decompresses a video from the custom binary format and returns a tensor.
        """
        compressed_files = sorted(os.listdir(compressed_path))
        reconstructed_frames = []
        
        iframe_file = [f for f in compressed_files if f.endswith('.jpg')][0]
        iframe_path = os.path.join(compressed_path, iframe_file)
        reconstructed_frame = self.to_tensor(Image.open(iframe_path).convert("RGB"))
        reconstructed_frames.append(reconstructed_frame)
        
        # Reconstruct motion vectors from indices
        shift_vectors = []
        k_range = self.motion_search_range
        for dy in [-k_range, 0, k_range]:
            for dx in [-k_range, 0, k_range]:
                shift_vectors.append(torch.tensor([dy, dx]))
        shift_vectors = torch.stack(shift_vectors)

        bin_files = sorted([f for f in compressed_files if f.endswith('.bin')])
        for i, file_name in tqdm(enumerate(bin_files)):
            frame_index = int(os.path.splitext(file_name)[0])
            
            with open(os.path.join(compressed_path, file_name), 'rb') as f:
                # 1. Read motion indices
                h_mv = np.frombuffer(f.read(2), dtype=np.uint16)[0]
                w_mv = np.frombuffer(f.read(2), dtype=np.uint16)[0]
                motion_indices_flat = np.frombuffer(f.read(h_mv * w_mv), dtype=np.uint8)
                motion_indices = torch.from_numpy(motion_indices_flat.reshape(h_mv, w_mv))
                motion_vectors = shift_vectors[motion_indices.long().reshape(-1)].reshape(h_mv, w_mv, 2)
                
                # 2. Read residual data
                num_entries = np.frombuffer(f.read(4), dtype=np.uint32)[0]
                huffman_codebook = {}
                for _ in range(num_entries):
                    symbol = np.frombuffer(f.read(2), dtype=np.int16)[0]
                    code_len = np.frombuffer(f.read(1), dtype=np.uint8)[0]
                    code = f.read(code_len).decode('ascii')
                    huffman_codebook[symbol] = code
                    
                num_bits = np.frombuffer(f.read(8), dtype=np.uint64)[0]
                byte_array = f.read()
                encoded_string = "".join(f"{byte:08b}" for byte in byte_array)[:num_bits]

                compressed_residuals = {
                    'huffman_codebook': huffman_codebook,
                    'encoded_string': encoded_string,
                    'original_shape': reconstructed_frame.shape
                }

            predicted_frame = torch.zeros_like(reconstructed_frame, device=self.device)
            reconstructed_frame_gpu = reconstructed_frame.to(self.device)
            k = self.block_size
            h, w = reconstructed_frame.shape[1], reconstructed_frame.shape[2]
            
            for r_idx in range(h_mv):
                for c_idx in range(w_mv):
                    mv = motion_vectors[r_idx, c_idx].to(self.device)
                    y_start, x_start = r_idx * k, c_idx * k
                    ref_y_start, ref_x_start = y_start + mv[0].item(), x_start + mv[1].item()
                    ref_y_start_c = np.clip(ref_y_start, 0, h - k)
                    ref_x_start_c = np.clip(ref_x_start, 0, w - k)
                    predicted_frame[:, y_start:y_start+k, x_start:x_start+k] = reconstructed_frame_gpu[:, ref_y_start_c:ref_y_start_c+k, ref_x_start_c:ref_x_start_c+k]

            decompressed_residuals = self.quantizer.decompress(compressed_residuals).to(self.device)
            reconstructed_frame = (predicted_frame + decompressed_residuals).clamp(0, 1).cpu()
            reconstructed_frames.append(reconstructed_frame)
        
        print("Decompression complete.")
        return torch.stack(reconstructed_frames)
