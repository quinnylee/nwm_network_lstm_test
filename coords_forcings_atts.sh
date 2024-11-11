csv_file="/home/qylee/Documents/NWM_access/runs/experiment_2024-10-07 15:31:08.815287/sample.csv"

command_template="python -m ngiab_data_cli -i ARG1,ARG2 -l -s"

tail -n +2 "$csv_file" | while IFS=',' read -r long lat _; do
	command_to_run=$(echo "$command_template" | sed "s/ARG1/$lat/" | sed "s/ARG2/$long/")
	echo "Executing: $command_to_run"
	$command_to_run
done
