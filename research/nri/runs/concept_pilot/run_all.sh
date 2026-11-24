cd /home/user/mnehmos.llm.playground/research/nri
xargs -P 4 -L 1 sh -c 'python -m sci.concept run $0 $1 runs/concept_pilot/wd$2_$0_s$1 --wd $2 > runs/concept_pilot/wd$2_$0_s$1.log 2>&1; echo "done $0 $1 $2"' < runs/concept_pilot/jobs.txt
echo ALLDONE
