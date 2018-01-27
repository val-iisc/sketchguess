#!/usr/bin/env bash
echo "Downloading model..."
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1kwDeK7qNzjxpCTAiCoKdPCCp7g-ntg9n' -O models/lstm_model_1lyr512hid_earlystopped.npz
echo "Downloading predictions..."
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=11fnnZJZ92qGn1ZX9V1kjtDAkz7U2IemQ' -O models/predictions_lstm_1lyr512hid.npz
