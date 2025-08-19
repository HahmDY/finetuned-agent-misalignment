domain=$1
model=$2

yaml="examples/train_full/${domain}_${model}.yaml"

conda activate llamafactory
set -e
llamafactory-cli train ${yaml}