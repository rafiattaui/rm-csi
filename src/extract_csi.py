import os
import glob
import struct
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import h5py

try:
    import dpkt
except ImportError:
    dpkt = None

try:
    from scapy.all import PcapReader
except ImportError:
    PcapReader = None

# =====================================================================
# Corrected Unpacker Function (Strict 32-bit Wraparound)
# =====================================================================

def unpack_float_nexmon(chip_type: int, length: int, payload: bytes) -> np.ndarray:
    """
    Unpacks Nexmon floating-point packed CSI payload for BCM4358 and BCM4366c0.
    Uses strict 32-bit bitmasks to avoid Python integer overflow.
    """
    Hout = np.zeros((length, 2), dtype=np.float64)
    
    for i in range(length):
        byte_idx = i * 4
        if byte_idx + 4 > len(payload):
            break
            
        # Parse 32-bit word (Big-Endian for Broadcom Nexmon)
        chunk = int.from_bytes(payload[byte_idx:byte_idx+4], byteorder='big', signed=False)
        
        # Extract 10-bit Real/Imag parts and 5-bit exponent
        i_val = chunk & 0x3FF
        q_val = (chunk >> 10) & 0x3FF
        exp = (chunk >> 20) & 0x1F
        
        # Sign-extend 10-bit signed integers (-512 to 511)
        if i_val & 0x200:
            i_val -= 0x400
        if q_val & 0x200:
            q_val -= 0x400
            
        # Apply exponent shift with 32-bit int truncation (& 0xFFFFFFFF)
        if exp > 0:
            shift = exp - 1
            # Perform bit-shift and wrap to 32-bit signed int
            i_shifted = (i_val << shift) & 0xFFFFFFFF
            q_shifted = (q_val << shift) & 0xFFFFFFFF
            
            # Convert back to signed int32 representation
            if i_shifted & 0x80000000:
                i_shifted -= 0x100000000
            if q_shifted & 0x80000000:
                q_shifted -= 0x100000000
                
            i_val = i_shifted
            q_val = q_shifted
            
        Hout[i, 0] = float(i_val)
        Hout[i, 1] = float(q_val)
        
    return Hout.flatten()


def read_pcap_packets(pcap_path: str, max_frames: int):
    """
    Reads raw packet payloads using dpkt (fast) or scapy (fallback).
    """
    packets = []
    if dpkt is not None:
        with open(pcap_path, 'rb') as f:
            pcap = dpkt.pcap.Reader(f)
            for k, (ts, buf) in enumerate(pcap):
                if k >= max_frames:
                    break
                packets.append(buf)
    elif PcapReader is not None:
        with PcapReader(pcap_path) as pcap:
            for k, pkt in enumerate(pcap):
                if k >= max_frames:
                    break
                packets.append(bytes(pkt))
    else:
        raise RuntimeError("Neither 'dpkt' nor 'scapy' is installed.")
    return packets


def parse_single_pcap(pcap_path: str, chip: str = '4366c0', bw: int = 80, max_frames: int = 100000):
    """
    Parses a single Nexmon PCAP file and extracts complex CSI array.
    """
    if bw == 20:
        non_zero = np.concatenate([np.arange(4, 32), np.arange(33, 61)])
    else:
        non_zero = np.concatenate([np.arange(6, 128), np.arange(131, 251)])

    num_subcarriers = int(bw * 3.2)
    expected_payload_len = num_subcarriers * 4

    csi_list = []
    seq_num = []
    core_num = []

    packets = read_pcap_packets(pcap_path, max_frames)

    for raw_pkt in packets:
        if len(raw_pkt) < expected_payload_len:
            continue

        # Target end payload section containing Nexmon CSI data
        payload = raw_pkt[-expected_payload_len - 60:]
        if len(payload) < expected_payload_len + 60:
            continue

        p14_bytes = payload[13*4 : 14*4]
        p14_hex = p14_bytes.hex()
        if len(p14_hex) < 8:
            continue

        core_num.append(p14_hex[0:2])
        seq_num.append(p14_hex[4:8])

        h_bytes = payload[15*4 : 15*4 + expected_payload_len]
        if len(h_bytes) < expected_payload_len:
            continue

        if chip in ['4339', '43455c0']:
            hout = np.frombuffer(h_bytes, dtype='>i2').astype(np.float64)
        elif chip == '4358':
            hout = unpack_float_nexmon(0, num_subcarriers, h_bytes)
        elif chip == '4366c0':
            hout = unpack_float_nexmon(1, num_subcarriers, h_bytes)
        else:
            raise ValueError(f"Invalid CHIP specified: {chip}")

        hout = hout.reshape(-1, 2)
        complex_csi = hout[:, 0] + 1j * hout[:, 1]
        csi_list.append(complex_csi)

    if not csi_list:
        return None, None, None

    csi_buff = np.array(csi_list)
    csi = csi_buff[:, non_zero]
    return csi, np.array(seq_num), np.array(core_num)


# =====================================================================
# Parallel Worker Function
# =====================================================================

def process_file_worker(args):
    pcap_path, output_dir, chip, bw, export_format = args
    file_stem = Path(pcap_path).stem
    out_path = os.path.join(output_dir, file_stem)
    
    try:
        csi, seq_num, core_num = parse_single_pcap(pcap_path, chip=chip, bw=bw)
        
        if csi is None:
            return pcap_path, False, "No valid CSI frames found."

        if export_format in ['npz', 'both']:
            np.savez_compressed(
                f"{out_path}_csi.npz",
                csi=csi,
                seq_num=seq_num,
                core_num=core_num
            )
            
        if export_format in ['h5', 'both']:
            with h5py.File(f"{out_path}_csi.h5", 'w') as hf:
                hf.create_dataset('csi_real', data=csi.real, compression="gzip")
                hf.create_dataset('csi_imag', data=csi.imag, compression="gzip")
                hf.create_dataset('csi_amplitude', data=np.abs(csi), compression="gzip")
                hf.create_dataset('csi_phase', data=np.angle(csi), compression="gzip")
                hf.create_dataset('seq_num', data=seq_num.astype('S'))
                hf.create_dataset('core_num', data=core_num.astype('S'))

        return pcap_path, True, f"Frames: {csi.shape[0]} | Subcarriers: {csi.shape[1]}"
        
    except Exception as e:
        return pcap_path, False, str(e)


# =====================================================================
# Main Batch Pipeline
# =====================================================================

def batch_convert_csi(
    input_dir: str, 
    output_dir: str, 
    chip: str = '4366c0', 
    bw: int = 80, 
    export_format: str = 'npz', 
    num_workers: int = 4
):
    os.makedirs(output_dir, exist_ok=True)
    pcap_files = glob.glob(os.path.join(input_dir, "**", "*.pcap"), recursive=True)
    
    if not pcap_files:
        print(f"No .pcap files found in '{input_dir}'")
        return

    print(f"Found {len(pcap_files)} PCAP files. Starting batch processing...")
    print(f"Workers: {num_workers} | Export format: {export_format.upper()}\n")

    tasks = [(f, output_dir, chip, bw, export_format) for f in pcap_files]

    successful = 0
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_file_worker, task): task[0] for task in tasks}
        
        for future in as_completed(futures):
            pcap_path, success, msg = future.result()
            filename = os.path.basename(pcap_path)
            
            if success:
                successful += 1
                print(f"[SUCCESS] {filename} -> {msg}")
            else:
                print(f"[FAILED]  {filename} -> Reason: {msg}")

    print(f"\nBatch processing completed: {successful}/{len(pcap_files)} converted successfully.")


if __name__ == "__main__":
    
    import logging
    logging.getLogger("scapy.runtime").setLevel(logging.ERROR)

    INPUT_FOLDER = 'data/HAR-3/CSI/M1'
    OUTPUT_FOLDER = '../data/HAR-3/CSI_processed'
    CHIP_TYPE = '4366c0'
    BANDWIDTH = 80
    EXPORT_FORMAT = 'npz'
    MAX_WORKERS = 4
    
    batch_convert_csi(
        input_dir=INPUT_FOLDER,
        output_dir=OUTPUT_FOLDER,
        chip=CHIP_TYPE,
        bw=BANDWIDTH,
        export_format=EXPORT_FORMAT,
        num_workers=MAX_WORKERS
    )