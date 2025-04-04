import os
import struct
import argparse
import torch
import numpy as np
from silero_vad import load_silero_vad, __version__ as silero_version

def convert_silero_vad(output_path, use_f16=True, sample_rate=16000):
    model = load_silero_vad()

    state_dict = model.state_dict()

    if sample_rate == 16000:
        model_prefix = "_model"
        input_channels = 129
        sr_suffix = "16k"
    elif sample_rate == 8000:
        model_prefix = "_model_8k"
        input_channels = 65
        sr_suffix = "8k"
    else:
        raise ValueError(f"Unsupported sample rate: {sample_rate}")

    base, ext = os.path.splitext(output_path)
    output_file = f"{base}-v{silero_version}_{sr_suffix}-ggml{ext}"

    print(f"Converting {sample_rate//1000}kHz model")
    print(f"Saving GGML Silero-VAD model to {output_file}")

    fout = open(output_file, "wb")

    # Write magic and version
    fout.write(struct.pack("i", 0x67676d6c))  # "ggml" in hex
    fout.write(struct.pack("i", 1))  # Version

    # Define and write the model architecture values
    fout.write(struct.pack("i", 1 if use_f16 else 0))  # Use f16 flag
    fout.write(struct.pack("i", sample_rate))  # Sample rate

    # Write dimensions for model
    n_encoder_layers = 4
    fout.write(struct.pack("i", n_encoder_layers))

    # Write encoder dimensions
    encoder_in_channels = [input_channels, 128, 64, 64]
    encoder_out_channels = [128, 64, 64, 128]
    kernel_size = 3

    for i in range(n_encoder_layers):
        fout.write(struct.pack("i", encoder_in_channels[i]))
        fout.write(struct.pack("i", encoder_out_channels[i]))
        fout.write(struct.pack("i", kernel_size))

    # Write LSTM dimensions
    lstm_input_size = 128
    lstm_hidden_size = 128
    fout.write(struct.pack("i", lstm_input_size))
    fout.write(struct.pack("i", lstm_hidden_size))

    # Write final conv dimensions
    final_conv_in = 128
    final_conv_out = 1
    fout.write(struct.pack("i", final_conv_in))
    fout.write(struct.pack("i", final_conv_out))

    # Helper function to write a tensor
    def write_tensor(name, tensor, f16=use_f16):
        print(f"  Writing {name} with shape {tensor.shape}")

        # Convert to numpy
        data = tensor.detach().cpu().numpy()

        # Convert to float16 if requested (and tensor is float32)
        if f16 and tensor.dtype == torch.float32:
            data = data.astype(np.float16)

        # Write tensor data
        data.tofile(fout)

    print("Writing model weights:")

    # 1. Encoder weights
    for i in range(n_encoder_layers):
        weight_key = f"{model_prefix}.encoder.{i}.reparam_conv.weight"
        bias_key = f"{model_prefix}.encoder.{i}.reparam_conv.bias"

        # Write conv weights and biases
        write_tensor(weight_key, state_dict[weight_key])
        write_tensor(bias_key, state_dict[bias_key])

    # 2. LSTM weights
    write_tensor("lstm_weight_ih", state_dict[f"{model_prefix}.decoder.rnn.weight_ih"])
    write_tensor("lstm_weight_hh", state_dict[f"{model_prefix}.decoder.rnn.weight_hh"])
    write_tensor("lstm_bias_ih", state_dict[f"{model_prefix}.decoder.rnn.bias_ih"])
    write_tensor("lstm_bias_hh", state_dict[f"{model_prefix}.decoder.rnn.bias_hh"])

    # 3. Final conv layer
    write_tensor("final_conv_weight", state_dict[f"{model_prefix}.decoder.decoder.2.weight"])
    write_tensor("final_conv_bias", state_dict[f"{model_prefix}.decoder.decoder.2.bias"])

    fout.close()
    print(f"Done! {sample_rate//1000}kHz model has been converted to GGML format: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Silero-VAD PyTorch model to GGML format")
    parser.add_argument("--output", type=str, required=True, help="Path to output GGML model file")
    parser.add_argument("--use-f16", action="store_true", help="Use float16 precision")
    parser.add_argument("--sample-rate", type=int, choices=[8000, 16000], default=16000, 
                        help="Sample rate: 8000 or 16000")

    args = parser.parse_args()
    convert_silero_vad(args.output, args.use_f16, args.sample_rate)
