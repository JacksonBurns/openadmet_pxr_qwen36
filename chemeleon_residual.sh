chemprop train \
    -s SMILES \
    --use-cuikmolmaker-featurization \
    -i predictions.csv \
    --from-foundation CheMeleon \
    --ffn-hidden-dim 512 \
    --ffn-num-layers 3 \
    --epochs 100 \
    --patience 10 \
    --batch-size 32 \
    --num-workers 4 \
    -o chemeleon_residual_output \
    --target-columns Ensemble \
    -t regression \
    -l mae \
    --metrics mse rmse r2 mae \
    --split random \
    --split-sizes 0.80 0.10 0.10 \
    --num-replicates 4 \
    --pytorch-seed 42 \
    --data-seed 42

chemprop predict \
    -s SMILES \
    -b 256 \
    -n 1 \
    -i test_predictions.csv \
    -o chemeleon_residual_preds.csv \
    --model-paths chemeleon_residual_output
