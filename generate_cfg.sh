# python bin/generate_cfg.py $1 $2
if [ "$#" -eq 1 ]; then
    python bin/generate_cfg.py $1
fi
if [ "$#" -eq 2 ]; then
    python bin/generate_cfg.py $1 $2
fi
if [ "$#" -eq 0 ]; then
    echo "Usage: ./generate_cfg.sh <filename> <funcname>"
fi