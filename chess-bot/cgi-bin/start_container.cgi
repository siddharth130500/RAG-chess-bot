#!/bin/bash
# Set PATH variable
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
echo "Content-type: text/html"
echo ""
/mnt/e/SDSC/Chess_bot/chess-bot/start_docker_container.sh
echo "<html><body><h1>Container started! Redirecting...</h1></body></html>"
echo "<script>setTimeout(function(){ window.location.href = 'http://localhost:5000/'; }, 3000);</script>"