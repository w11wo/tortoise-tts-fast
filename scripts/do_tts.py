#!/usr/bin/env python3

import torchaudio
from simple_parsing import ArgumentParser

from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_voice


if __name__ == "__main__":
    parser = ArgumentParser(
        description="TorToiSe is a text-to-speech program that is capable of synthesizing speech "
        "in multiple voices with realistic prosody and intonation."
    )
    parser.add_argument(
        "--ar_checkpoint",
        required=False,
        default=None,
        help="Specify path to model checkpoint",
    )
    parser.add_argument("--text", required=True, help="Specify input text")
    parser.add_argument("--preset", required=True, help="Specify desired audio quality")
    parser.add_argument("--half", type=bool, default=False, help="Use half precision")
    parser.add_argument(
        "--voice", default=None, help="Specify audio directory for conditioning latents"
    )
    parser.add_argument(
        "--output_name", default="generated.wav", help="Generated audio file name"
    )

    args = parser.parse_args()

    tts = TextToSpeech(ar_checkpoint=args.ar_checkpoint)
    voice_samples, conditioning_latents = None, None

    if args.voice:
        voice_samples, conditioning_latents = load_voice(args.voice)

    gen = tts.tts_with_preset(
        args.text,
        voice_samples=voice_samples,
        conditioning_latents=conditioning_latents,
        preset=args.preset,
        half=args.half,
    )

    torchaudio.save(args.output_name, gen.squeeze(0).cpu(), 24000)
