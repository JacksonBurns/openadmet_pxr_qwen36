chemprop train \
    -s SMILES \
    --use-cuikmolmaker-featurization \
    -i tdc_pampa_ncats.csv \
    --from-foundation CheMeleon \
    --ffn-hidden-dim 512 \
    --ffn-num-layers 3 \
    --epochs 100 \
    --patience 5 \
    --batch-size 32 \
    --num-workers 4 \
    -o chemeleon_pampa_output \
    --target-columns Y \
    -t classification \
    --metrics roc prc accuracy f1 \
    --split KMEANS \
    --split-sizes 0.80 0.20 0.00 \
    --num-replicates 4 \
    --pytorch-seed 42 \
    --data-seed 42

# loop through train.csv, test_phase1.csv, and test.csv for prediction
for dset in train test_phase1 test; do
     chemprop predict \
        -s SMILES \
        -b 256 \
        -n 1 \
        -i ${dset}.csv \
        -o ${dset}_chemeleon_pampa_preds.csv \
        --model-paths chemeleon_pampa_output \
        --drop-extra-columns
done

python chemeleon_pampa_postprocess.py
