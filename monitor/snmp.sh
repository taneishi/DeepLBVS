#!/bin/sh
# */10 * * * * ~/work/monitor/snmp.sh >> ~/work/monitor/snmp.log

COMMUNITY=administrator
#IPADDRESS=192.168.100.5
IPADDRESS=10.240.8.236

snmpwalk -v 2c -c $COMMUNITY $IPADDRESS iso.3.6.1.4.1.21317.1.3.2.2.3.4.8.2.1.0
snmpwalk -v 2c -c $COMMUNITY $IPADDRESS iso.3.6.1.4.1.21317.1.3.2.2.3.4.8.2.2.0
for i in $(seq 1 8)
do snmpwalk -v 2c -c $COMMUNITY $IPADDRESS iso.3.6.1.4.1.21317.1.3.2.2.2.2.1.1.2.$i
done
