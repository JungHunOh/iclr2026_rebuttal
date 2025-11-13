
target_user="stella"  # The user to wait for
kill_user="dh6dh"               # The user whose python processes you want to kill

while true; do
    echo "Waiting for $target_user to log in..."
    pids=$(ps -u "$kill_user" -o pid=,comm= | awk '$2 ~ /^python/ {print $1}')
    echo "Current Processes are $pids"
    if who | grep -q "^$target_user\b"; then
        echo "$target_user has logged in. Killing all python processes of $kill_user..."

        pids=$(ps -u "$kill_user" -o pid=,comm= | awk '$2 ~ /^python/ {print $1}')

        if [ -z "$pids" ]; then
            echo "No python processes found for user $kill_user."
        else
            for pid in $pids; do
                echo "Killing python process $pid"
                kill -9 "$pid"
            done
            echo "All python processes of $kill_user killed."
        fi

        break  # exit after killing
    fi

    sleep 2  # check every 10 seconds
done