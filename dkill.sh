kill $(ps j | grep $1 | awk '{print $2}')
