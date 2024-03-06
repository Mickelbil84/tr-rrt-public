TEST=1

python scripts/run_tests.py --name="abc_light" --num_test=$TEST --gui=False
python scripts/run_tests.py --name="aj" --num_test=$TEST --gui=False
python scripts/run_tests.py --name="alpha" --num_test=$TEST --gui=False
 python scripts/run_tests.py --name="az" --num_test=$TEST --gui=False

python scripts/run_tests.py --name="doublealpha" --num_test=$TEST --gui=False
python scripts/run_tests.py --name="duet-g1" --num_test=$TEST --gui=False
python scripts/run_tests.py --name="key_light_1" --num_test=$TEST --gui=False
python scripts/run_tests.py --name="mobius_light" --num_test=$TEST --gui=False

python scripts/run_tests.py --name="06397" --num_test=$TEST --gui=False
python scripts/run_tests.py --name="07631" --num_test=$TEST --gui=False
python scripts/run_tests.py --name="08862" --num_test=$TEST --gui=False
python scripts/run_tests.py --name="09301" --num_test=$TEST --gui=False

python scripts/run_tests.py --name="11019" --num_test=$TEST --gui=False
python scripts/run_tests.py --name="11356" --num_test=$TEST --gui=False --sampling=5000
python scripts/run_tests.py --name="11780" --num_test=$TEST --gui=False
python scripts/run_tests.py --name="16505" --num_test=$TEST --gui=False
