#!/usr/bin/env python
##
##  unagi.py, a system monitoring tool
##
##  Copyright (c) 2005  Yusuke Shinyama <yusuke at cs dot nyu dot edu>
##
##  Permission is hereby granted, free of charge, to any person
##  obtaining a copy of this software and associated documentation
##  files (the "Software"), to deal in the Software without
##  restriction, including without limitation the rights to use,
##  copy, modify, merge, publish, distribute, sublicense, and/or
##  sell copies of the Software, and to permit persons to whom the
##  Software is furnished to do so, subject to the following
##  conditions:
##
##  The above copyright notice and this permission notice shall be
##  included in all copies or substantial portions of the Software.
##
##  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY
##  KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
##  WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
##  PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
##  COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
##  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
##  OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
##  SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
##

import os,sys
import re
import socket
import asyncore
import signal
import select
import struct
import errno

# your network information
SIGNATURE = ''
P2P_PORT = 10080                # >1024
HTTP_PORT = 8000                # >1024
P2P_SCAN_RANGES = ['127.0.0.1'] # [ '192.168.0.1-99' ]
P2P_ALLOW_RANGES = ['127.0.0.'] # [ '192.168.' ]
HTTP_ALLOW_RANGES = ['127.']    # [ '192.168.', '127.' ]
UNAGI_USER = ''                 # 'unagi', make sure this included in /etc/passwd entries.

# interval settings
UPDATE_INTERVAL_SECONDS = 600   # 1 interval = 600secs (10mins).
UPDATE_INTERVAL_SECONDS = 10    # 1 interval = 600secs (10mins).
BROADQUERY_INTERVAL_COUNT = 3   # broadcast addresses every 3 intervals (30mins)
DOWN_COUNT = 3                  # if a machine doesn't respond for 3 intervals (30mins),
                                # mark it as "down".
REMOVAL_COUNT = 1440            # if a machine doesn't respond for 1440 intervals (10days),
                                # remove it from the database.
HISTORY_SIZE = 12               # the number of entries to preserve in the database.

# OS-specific parameters
MAX_PROCESS_ENTRIES = 5         # max process entries to obtain
MAX_USER_ENTRIES = 10           # max user entries to obtain
MAX_MESSAGE_ENTRIES = 5         # max syslog entries to obtain at once
MAX_MESSAGE_BUFFSIZE = 10       # max syslog entries to preserve
TIME_UNIT = 60                  # duration to calc average (60secs)

# HTML table colors
BGCOLOR_NORMAL    = "#ddddff"
BGCOLOR_DOWN      = "#888888"
BGCOLOR_LOW_CPU   = "#88ff88"
BGCOLOR_HIGH_CPU  = "#ffffaa"
BGCOLOR_LOW_DISK  = "#ffcccc"
BGCOLOR_HIGH_DISK = "#ff8888"

STAT1 = '''<a href="#{0}" onclick="toggle1('i{1}-{2}');">{3}%CPU, {4}users</a>
<div id="i{5}-{6}" style="display:none;">
<p><b>Resources</b><br><table class="c2">
<tr><td>CPU</td><td colspan="2" align=center>User:{7}%, Sys:{8}%, Idle:{9}%</td></tr>
<tr><td>Memory</td><td colspan="2" align=center>Used:{10}%, Swapped:{11}%</td></tr>
<tr><td>Pages</td><td colspan="2" align=center>Active:{12}%, Inactive:{13}%</td></tr>
<tr><td>LoadAvg</td><td colspan="2" align=center>{14}.2f, {15}.2f, {16}.2f</td></tr>
<tr><td>Swap</td><td align=center>{17}% in / {18}s</td><td align=center>{19}% out / {20}s</td></tr>
<tr><td>Disk</td><td align=center>{21}M in / {22}s</td><td align=center>{23}M out / {24}s</td></tr>
<tr><td>Network</td><td align=center>{25}M in / {26}s</td><td align=center>{27}M out / {28}s</td></tr></table>'''

# version info
VERSION = 'unagi.py/0.38'

##  misc utilities.
def split2(sep, s):
    if not s:
        return []
    else:
        return s.split(sep)

R = re.compile(r"[\000-\037\176-\377]")
S = re.compile(r"\s+")
def clean(s):
    return S.sub(" ", R.sub("?", s.strip()))

# abbrev.
def showtime(d):
    d = int(d/60)
    if d < 60:
        return '%dm' % d
    elif d < 60*24:
        return '%dh%dm' % divmod(d, 60)
    else:
        return '%dd%dh' % divmod(d/60, 24)

def safeint(x):
    try:
        return int(x)
    except ValueError:
        return -1

def getfilesize(fname):
    import stat
    try:
        return os.stat(fname)[stat.ST_SIZE]
    except OSError:
        return 0

def getcurtime():
    import time
    return int(time.time())

def getcmdout(cmdline):
    import popen2
    p = popen2.Popen3(cmdline, 0)
    p.tochild.close()
    lines = p.fromchild.readlines()
    p.fromchild.close()
    if p.wait() != 0:       # error?
        return []
    return lines

ISO8859CHARS = { '&':'&amp;', '>':'&gt;', '<':'&lt;' }
for i in range(127, 256):
    ISO8859CHARS[chr(i)] = '&#%d;' % i
def htmlquote(s):
    return ''.join(map(lambda c:ISO8859CHARS.get(c,c), s))

# check if the addr is in addranges.
REP = re.compile(r"^(\d+)-(\d+)$")
def inrange(addr, addranges):
    a0 = addr.split('.')
    for r in addranges:
        if addr == r or addr[:len(r)] == r: return 1
        a1 = r.split('.')
        if len(a1) == 4:
            m = REP.match(a1[3])
            if m and safeint(m.group(1)) <= int(a0[3]) and int(a0[3]) <= safeint(m.group(2)): return 1
    return 0

#assert inrange('1.2.3.4', ['1.2.3.4'])
#assert inrange('1.2.3.4', ['1.2'])
#assert inrange('1.2.3.4', ['1.2.3.1-5'])
#assert not inrange('1.2.3.4', ['1.2.3.5'])
#assert not inrange('1.2.3.4', ['1.3'])
#assert not inrange('1.2.3.4', ['1.2.3.5-10'])  

# convert address ranges into list of addresses.
def convranges(addranges):
    addrs = []
    for addr in addranges:
        a1 = addr.split('.')
        assert 3 <= len(a1), 'invalid address range: %s' % addr
        head = '.'.join(a1[:3])
        if len(a1) == 3 or not a1[3]:
            for i in range(1,255):
                addrs.append('%s.%d' % (head, i))
        else:
            m = REP.match(a1[3])
            if m:
                for i in range(int(m.group(1)), int(m.group(2))+1):
                    addrs.append('%s.%d' % (head, i))
            else:
                addrs.append(addr)
    return addrs

assert convranges(['1.2.3.4']) == ['1.2.3.4']
assert convranges(['1.2.3.4-6']) == ['1.2.3.4','1.2.3.5','1.2.3.6']
assert len(convranges(['1.2.3'])) == 254
assert len(convranges(['1.2.3.'])) == 254

##  StatusReporter
class StatusReporter:

    def __init__(self):
        self.buffer = None
        self.last_modified = 0
        self.report_sent = {}
        self.name = socket.gethostname()
        return

    def __repr__(self):
        return '<StatusReporter(%s): %s>' % (self.name, repr(self.buffer))

    def report_updated(self, buffer):
        self.last_modified = getcurtime()
        self.buffer = buffer
        self.report_sent = {}
        return

    def send_report(self, addr):
        # check double sent
        if not self.buffer or self.report_sent.has_key(addr):
            return ''
        self.report_sent[addr] = 1
        return self.buffer

##  LinuxStatusReporter
class LinuxStatusReporter(StatusReporter):

    def __init__(self):
        assert sys.platform[:5] == "linux", 'sys.platform doesn\'t begin with "linux2".'
        StatusReporter.__init__(self)
        self.msgspos = getfilesize(self.MESSAGES_PATHNAME)
        self.session = {}
        (self.tuser0, self.tnice0, self.tsys0, self.tidle0,
         self.sin0, self.sout0, self.bin0, self.bout0,
         self.rx0, self.tx0) = (0,0,0,0,0,0,0,0,0,0)
        return

    def __repr__(self):
        return '<LinuxStatusReporter(%s): %s>' % (self.name, repr(self.buffer))

    # read system status
    PAT_UPTIME = re.compile(r"([0-9]+)")
    PAT_LOADAVG = re.compile(r"([0-9]+)\.([0-9][0-9]) +([0-9]+)\.([0-9][0-9]) +([0-9]+)\.([0-9][0-9])")
    PAT_MEMINFO = re.compile(r"([a-zA-Z0-9]+): *([0-9]+)", re.I)
    PAT_MEMINFO_MEMFREE = re.compile(r"memfree: *([0-9]+)", re.I)
    PAT_MEMINFO_BUFF = re.compile(r"buffers: *([0-9]+)", re.I)
    PAT_MEMINFO_CACHED = re.compile(r"cached: *([0-9]+)", re.I)
    PAT_MEMINFO_SWAPTOTAL = re.compile(r"swaptotal: *([0-9]+)", re.I)
    PAT_MEMINFO_SWAPFREE = re.compile(r"swapfree: *([0-9]+)", re.I)
    PAT_MEMINFO_ACTIVE = re.compile(r"active: *([0-9]+)", re.I)
    PAT_MEMINFO_INACTIVE = re.compile(r"inactive: *([0-9]+)", re.I)
    PAT_STAT_CPU = re.compile(r"cpu +([0-9]+) +([0-9]+) +([0-9]+) +([0-9]+)", re.I)
    PAT_STAT_DISKIO = re.compile(r"disk_io: *(.+)", re.I)
    PAT_DISKIO1 = re.compile(r"\([0-9,]+\):\([0-9]+,[0-9]+,([0-9]+),[0-9]+,([0-9]+)\)")
    PAT_STAT_SWAP = re.compile(r"swap +([0-9]+) +([0-9]+)", re.I)
    PAT_NET_DEV = re.compile(r" *eth[0-9]+: *([0-9]+)"+r" +[0-9]+"*7+r" +([0-9]+)", re.I)
    DISK_UNIT = 2*1024          # disk I/O traffic (2048blocks = 1MBytes)
    NET_UNIT = 1024*1024        # network traffic (1MBytes)
    def get_info(self):
        dtime = getcurtime() - self.last_modified
        # hostname, curtime
        info = [ self.name, getcurtime(), TIME_UNIT ]
        # uptime
        fp = open("/proc/uptime")
        m = self.PAT_UPTIME.match(fp.readline())
        info.append(m.group(0))
        fp.close()
        # loadavg1, loadavg2, loadavg3
        fp = open("/proc/loadavg")
        m = self.PAT_LOADAVG.match(fp.readline())
        info.extend( [m.group(1)+m.group(2), m.group(3)+m.group(4), m.group(5)+m.group(6)] )
        fp.close()
        # mpused, mpswaped, mpactive, mpinactive
        fp = open("/proc/meminfo")
        dic = {}
        for line in fp.readlines():
            m = self.PAT_MEMINFO.match(line)
            if m:
                dic[m.group(1).lower()] = int(m.group(2))
        fp.close()
        mtotal = dic.get('memtotal', 1)
        info.append((mtotal-(dic.get('memfree',0)+dic.get('buffers',0)+dic.get('cached',0)))*100/mtotal)
        info.append((dic.get('swaptotal',0)-dic.get('swapfree',0))*100/mtotal)
        info.append(dic.get('active',0)*100/mtotal)
        info.append(dic.get('inactive',0)*100/mtotal)
        # tusr, tsys, tidl, sin, sout, bin, bout
        fp = open("/proc/stat")
        (tuser, tnice, tsys, tidle, sin, sout, bin, bout) = (0, 0, 0, 0, 0, 0, 0, 0)
        for line in fp.readlines():
            m = self.PAT_STAT_CPU.match(line)
            if m:
                (tuser, tnice, tsys, tidle) = map(long, m.groups())
                continue
            m = self.PAT_STAT_SWAP.match(line)
            if m:
                (sin, sout) = map(long, m.groups())
                continue
            m = self.PAT_STAT_DISKIO.match(line)
            if m:
                for s in m.group(1).split(" "):
                    m = self.PAT_DISKIO1.match(s)
                    if m:
                        bin = bin + long(m.group(1))
                        bout = bout + long(m.group(2))
                continue
        fp.close()
        if not self.last_modified:
            info.extend( [0,0,0,0,0,0,0] )
        else:
            (p1,p2,p3) = (tuser-self.tuser0 + tnice-self.tnice0, tsys-self.tsys0, tidle-self.tidle0)
            pp = max(1, p1+p2+p3)
            (r1,r2) = (p1*100/pp, p2*100/pp)
            info.extend(map(int, [r1, r2, 100-r1-r2,
                    (sin-self.sin0)*TIME_UNIT*100/(dtime*mtotal),
                    (sout-self.sout0)*TIME_UNIT*100/(dtime*mtotal),
                    (bin-self.bin0)*TIME_UNIT/(dtime*self.DISK_UNIT),
                    (bout-self.bout0)*TIME_UNIT/(dtime*self.DISK_UNIT)
                    ]))
        (self.tuser0, self.tnice0, self.tsys0, self.tidle0,
                self.sin0, self.sout0, self.bin0, self.bout0) = (tuser, tnice, tsys, tidle, sin, sout, bin, bout)
        # drx, dtx
        fp = open("/proc/net/dev")
        (rx, tx) = (0, 0)
        for line in fp.readlines()[2:]:
            m = self.PAT_NET_DEV.match(line)
            if m:
                rx = rx + long(m.group(1))
                tx = tx + long(m.group(2))
        if not self.last_modified:
            info.extend([ 0, 0 ])
        else:
            info.extend([ int((rx-self.rx0)*TIME_UNIT/(dtime*self.NET_UNIT)),
                                        int((tx-self.tx0)*TIME_UNIT/(dtime*self.NET_UNIT)) ])
        (self.rx0, self.tx0) = (rx, tx)
        fp.close()
        return ' '.join(map(str, info))

    # get processes
    PS_CMDLINE = "ps -A --no-headers --sort -%cpu --format 'sess user %cpu %mem stat time comm'"
    def get_proc(self):
        procs = []
        self.session = {}
        for line in getcmdout(self.PS_CMDLINE):
            # (sess, username, pcpu, pmem, stat, time, command)
            flds = clean(line).split(" ")
            if len(flds) == 7:
                self.session[safeint(flds[0])] = flds[6].replace(" ", "_")
                if flds[3] != "0.0" and len(procs) < MAX_PROCESS_ENTRIES: # 0.0 < pcpu
                    procs.append(' '.join(flds[1:]))
        return '\t'.join(procs)

    # get users
    UTMP_PATHNAME = "/var/run/utmp"
    UTMP_RECORD_SIZE = 384
    UTMP_RECORD_FORMAT = '< H 2x L 32x 4x 32s 256x 4x 4x L 40x' # (ut_type, ut_pid, ut_user, ut_tv_sec)
    UTMP_UT_USERPROC = 7
    def get_user(self):
        try:
            fp = open(self.UTMP_PATHNAME, "rb")
        except IOError, e:
            sys.stderr.write("Cannot open the utmp file %s, reason: %s\n" % (self.UTMP_PATHNAME, e))
            return ""
        fs = getfilesize(self.UTMP_PATHNAME)
        if (fs % self.UTMP_RECORD_SIZE) != 0:        # wrong record size?
            fp.close()
            return ""
        users = []
        for i in xrange(fs / self.UTMP_RECORD_SIZE):
            s = fp.read(self.UTMP_RECORD_SIZE)
            (ut_type, ut_pid, ut_user, ut_tv_sec) = struct.unpack(self.UTMP_RECORD_FORMAT, s)
            if ut_type != self.UTMP_UT_USERPROC: continue
            users.append("%s -%s %s" % (ut_user.replace("\x00", ""),
                                            showtime(getcurtime()-ut_tv_sec),
                                            self.session.get(ut_pid, "?")))
            if MAX_USER_ENTRIES <= len(users): break
        fp.close()
        return '\t'.join(users)

    # get /var/log/messages
    MESSAGES_PATHNAME = "/var/log/syslog"
    def get_mesg(self):
        try:
            fp = open(self.MESSAGES_PATHNAME)
        except IOError, e:
            sys.stderr.write("Cannot open the messages file %s, reason: %s\n" % (self.MESSAGES_PATHNAME, e))
            return ""
        # read the file from the point last read.
        fs = getfilesize(self.MESSAGES_PATHNAME)
        if fs < self.msgspos:
            # now the file is truncated.
            self.msgspos = 0
        fp.seek(self.msgspos)
        msgs = []
        while 1:
            line = fp.readline()
            if not line: break
            line = clean(line)
            if line:
                msgs.append(line)
                if MAX_MESSAGE_ENTRIES < len(msgs):
                    msgs = msgs[:MAX_MESSAGE_ENTRIES]
        self.msgspos = fp.tell()
        fp.close()
        return '\t'.join(msgs)

    def update_report(self):
        # get kernel info.
        # (hostname, curtime, uptime, loadavg1, loadavg2, loadavg3,
        #  mpused, mpswaped, mpactive, mpinactive,
        #  tuser, tsys, tidle, sin, sout, bin, bout, drx, dtx)
        # combine them into one
        records = (self.get_info(), self.get_proc(),
                             self.get_user(), self.get_mesg())
        self.report_updated('\n'.join(records))
        return

##  Host
class IllegalStatusError(ValueError):
    pass

class Host:
    def __init__(self, addr):
        self.addr = addr
        self.name = ''
        self.history = []
        self.messages = []
        self.count = 0
        self.uptime = 0
        self.curtime = 0
        self.alive = 1
        return
    
    def __repr__(self):
        return '<Host: %s>' % self.addr

    def mark_down(self):
        if self.history and self.history[0] != None:
            self.history.insert(0, None)
        return

    def check_down(self):
        self.count = self.count + 1
        if self.alive and DOWN_COUNT <= self.count:
            self.mark_down()
            self.alive = 0
        return REMOVAL_COUNT < self.count

    def update_status(self, status, histsize):
        self.count = 0
        self.alive = 1
        if not status:
            # just ping
            return
        flds = status.split("\n")
        if len(flds) != 4:
            raise IllegalStatusError('status:'+repr(status))
        (raw_info, raw_proc, raw_user, raw_mesg) = flds
        st_info = split2(" ", raw_info)
        if len(st_info) != 20:
            raise IllegalStatusError('raw_info:'+repr(raw_info))
        st_proc = split2("\t", raw_proc)[:MAX_PROCESS_ENTRIES]
        if filter(lambda x: len(x.split(" ")) != 6, st_proc):
            raise IllegalStatusError('raw_proc:'+repr(raw_proc))            
        st_user = split2("\t", raw_user)[:MAX_USER_ENTRIES]
        if filter(lambda x: len(x.split(" ")) != 3, st_user):
            raise IllegalStatusError('raw_user:'+repr(raw_user))            
        st_mesg = split2("\t", raw_mesg)[:MAX_MESSAGE_ENTRIES]
        if st_mesg:
            self.messages.append(st_mesg)
            if MAX_MESSAGE_BUFFSIZE < len(self.messages):
                self.messages = self.messages[:MAX_MESSAGE_BUFFSIZE]
        if not self.name:
            self.name = st_info[0]
        uptime = safeint(st_info[3])
        if uptime < self.uptime:
            # if the uptime is less than the previous one, we consider it was down.
            self.mark_down()
        self.uptime = uptime
        self.curtime = safeint(st_info[1])
        # insert newest entry
        self.history.insert(0, (st_info, st_proc, st_user, st_mesg))
        # remove old entries
        if histsize < len(self.history):
            self.history = self.history[:histsize]
        return
    
    def make_html_report1(self, out, hid, port):
        nents = len(self.history)
        if not nents: return
        raw1 = []
        raw2 = []
        t = 0
        for status in self.history:
            if not status:
                # host is down.
                raw1.append('<td>(down)</td>\n')
                raw2.append('<td bgcolor="%s"></td>\n' % BGCOLOR_DOWN)
            else:
                # host is up.
                (st_info, st_proc, st_user, st_mesg) = status
                # disp info
                (curtime, timeunit, uptime, 
                        loadavg1, loadavg2, loadavg3,
                        mpused, mpswaped, mpactive, mpinactive,
                        tuser, tsys, tidle, sin, sout, bin, bout, drx, dtx) = map(safeint, st_info[1:])
                stat1 = STAT1.format(
                        self.name, hid, t, '%02d' % (tuser+tsys), len(st_user),
                        hid, t,
                        tuser, tsys, tidle,
                        mpused, mpswaped,
                        mpactive, mpinactive,
                        '%.2f' % (loadavg1/100.0), '%.2f' % (loadavg2/100.0), '%.2f' % (loadavg3/100.0),
                        sin, timeunit, sout, timeunit,
                        bin, timeunit, bout, timeunit,
                        drx, timeunit, dtx, timeunit)
                # disp proc
                stat1 = stat1 + '''\n<p><b>Processes</b><br><table class="c2">
<tr><th>user</th><th>cpu/mem</th><th>stat</th><th>time</th><th>cmd</th></tr>
<tr>'''
                for s in st_proc:
                    # (username, pcpu, pmem, pstat, ptime, command)
                    flds = s.split(" ")
                    if len(flds) == 6:
                        stat1 = stat1 + '<tr><td>%s</td><td>%s%%/%s%%</td><td>%s</td><td>%s</td><td>%s</td></tr>' % tuple(flds)
                stat1 = stat1 + '</table>'
                # disp user
                stat1 = stat1 + '''\n<p><b>Users</b><br><table class="c2">
<tr><th>user</th><th>login</th><th>what</th></tr>
<tr>'''
                for s in st_user:
                    # (username, tty, linefrom, idle, what)
                    flds = s.split(" ")
                    if len(flds) == 3:
                        stat1 = stat1 + '<tr><td>%s</td><td>%s</td><td>%s</td></tr>' % \
                                        tuple(flds)
                stat1 = stat1 + '</table>'
                # disp mesg
                stat1 = stat1 + '\n<p><b>Messages</b><br>%s<br>' % \
                                ('<br>'.join(st_mesg))
                # set bgcolor
                bgcolor = BGCOLOR_NORMAL
                if 40 <= bin+bout:
                    bgcolor = BGCOLOR_HIGH_DISK
                elif 20 <= bin+bout:
                    bgcolor = BGCOLOR_LOW_DISK
                if 90 <= tuser+tsys:
                    bgcolor = BGCOLOR_HIGH_CPU
                elif 45 <= tuser+tsys:
                    bgcolor = BGCOLOR_LOW_CPU
                # disp column
                msgent = ''
                if st_mesg:
                    msgent = ' <font color=red>[%d]</font>' % len(st_mesg)
                raw1.append('<td>-%s%s</td>' % (showtime(getcurtime()-curtime), msgent))
                raw2.append('<td bgcolor="%s">%s</td>\n' % (bgcolor, stat1))
            t = t + 1
        #
        out.append('<p><a name="%s"></a><a href="http://%s:%d/"><b>@%s</b></a><small> (%s) up: %s \n' % \
                             (self.name, self.addr, port, self.name.upper(), self.addr, showtime(self.uptime)))
        out.append('<a href="#%s" onclick="toggleall(\'i%d\',%d)">[Expand All]</a>\n' % (self.name, hid, nents))
        out.append('<a href="#%s" onclick="toggle1(\'a%d\')">[Expand Messages]</a>\n' % (self.name, hid))
        out.append('</small><table class="c1" border="1" cellspacing="0">\n')
        out.extend(raw1)
        out.append('</tr><tr valign=top>\n')
        out.extend(raw2)
        out.append('</tr></table>\n<div class="c3" id="a%s" style="display:none;">' % hid + \
                             '<b>Messages@%s:</b><br>%s<br></div>\n' % \
                             (self.name.upper(),
                                '<br>'.join(map(lambda msg1:'<br>'.join(msg1), self.messages))))
        return

##  HostDB
class HostDB:
    def __init__(self, scan_ranges, history_size, update_interval):
        self.hostdic = {}
        self.name = socket.gethostname()
        self.scanaddrs = convranges(scan_ranges)
        self.history_size = history_size
        self.last_updated = 0
        self.update_interval = update_interval
        sys.stderr.write("HostDB initialized: scan: %s\n" % ', '.join(scan_ranges))
        return

    def __repr__(self):
        return '<HostDB(%s): %s>' % (self.name, repr(self.hostdic.values()))

    def allhosts(self):
        return self.hostdic.values()

    def needscan(self):
        # if I find at least two peers (including myself), stop scanning.
        return len(self.hostdic) < 2 and self.scanaddrs

    def nextscan(self):
        return self.scanaddrs.pop()

    def intern_host(self, addr):
        try:
            host = self.hostdic[addr]
        except KeyError:
            host = Host(addr)
            sys.stderr.write("HostDB: added: %s\n" % host)
            self.hostdic[addr] = host
        return host

    def remove_host(self, addr):
        if addr in self.hostdic:
            sys.stderr.write("HostDB: removed: %s\n" % self.hostdic[addr])
            del self.hostdic[addr]
        return

    def receive_host_status(self, addr, status):
        try:
            self.intern_host(addr).update_status(status, self.history_size)
        except IllegalStatusError, x:
            sys.stderr.write("IllegalStatusError: %s\n" % x)
        return

    def update_all_hosts(self):
        removed = []
        for host in self.hostdic.values():
            # not responding for longtime?
            if host.check_down():
                removed.append(host)
        for host in removed:
            self.remove_host(host.addr)
        self.last_updated = getcurtime()
        return

    def make_html_report(self, out, port):
        import time
        title = 'Status@%s' % self.name.upper()
        out.append('<html><head><meta http-equiv="refresh" content="%d; URL=/">\n' % self.update_interval)
        out.append('<title>%s</title>\n' % title)
        out.append('''<script language="javascript">
<!--
function is_shown(x) {
 e=document.getElementById(x);
 if (e) {
    return e.style.display == "block";
 } else {
    return false;
 }
}
function show(x,y) {
 e=document.getElementById(x);
 if (e) {
    if (y) {
     e.style.display="block";
    } else {
     e.style.display="none"; }
    }
}
function toggle1(x) {
 show(x, !is_shown(x));
}
function toggleall(z,n) {
 x = !is_shown(z+"-0");
 for(i=0; i<n; i++) { show(z+"-"+i, x); }
}
// -->
</script>
<style>
<!--
body { background-color:white; }
table.c1 { font-size:80%; }
table.c2 { font-size:80%; }
div.c3 { font-size:80%; padding:1px; border:1px solid black; background-color:#ffffcc; }
a:hover { color: red; }
// -->
</style></head><body>
''')
        out.append('<h1>%s</h1>\n' % title)
        out.append('''<p>Last updated: %s (-%s, every %d seconds)
<p><small><strong>Click each cell to expand details. (Javascript required)</strong></small>
<hr noshade size="2">\n''' % \
                             (time.ctime(self.last_updated),
                                showtime(getcurtime()-self.last_updated),
                                self.update_interval))
        hosts = self.hostdic.values()
        hosts.sort(lambda h1,h2: cmp(h1.name, h2.name))
        hid = 0
        for host in hosts:
            host.make_html_report1(out, hid, port)
            hid = hid + 1
        out.append('<hr noshade size="2"><table class="c1"><tr>%s</tr></table>' %
                             ''.join(map(lambda (col,legend):
                                     '<td bgcolor="%s" width="12"></td><td>%s &nbsp;</td>' % (col,legend),
                                     ((BGCOLOR_NORMAL,"Normal"), (BGCOLOR_DOWN,"Down"),
                                      (BGCOLOR_LOW_CPU,"Low CPU"), (BGCOLOR_HIGH_CPU,"High CPU"),
                                      (BGCOLOR_LOW_DISK,"Low disk"), (BGCOLOR_HIGH_DISK,"High disk")))))
        out.append('<div align=right><address>%s</address>\n</body></html></div>\n' % VERSION)
        return

##  ASyncP2PServer
p2pserv = None
class ASyncP2PServer(asyncore.dispatcher):
    def __init__(self, update_interval, hostdb, reporter, signature, allow_ranges, p2p_port, bindaddr="0.0.0.0", debug=0):
        global p2pserv
        asyncore.dispatcher.__init__(self)
        self.signature = signature
        self.debug = debug
        self.p2p_port = p2p_port
        self.reporter = reporter
        self.hostdb = hostdb
        self.tickcount = 0
        self.update_interval = update_interval
        self.allow_ranges = allow_ranges
        self.create_socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.bind((bindaddr, p2p_port))
        sys.stderr.write("ASyncP2PServer: allow: %s\n" % ', '.join(allow_ranges))
        sys.stderr.write("ASyncP2PServer: listening: %s:%d\n" % (bindaddr, p2p_port))
        p2pserv = self
        def sigfunc(signum, frame):
            global p2pserv
            p2pserv.scan()
            return
        signal.signal(signal.SIGALRM, sigfunc)
        return
    
    def handle_connect(self):
        return
    
    def writable(self):
        return self.hostdb.needscan()
    
    def handle_write(self):
        addr = self.hostdb.nextscan()
        self.sendto(addr, "?")
        if not self.hostdb.needscan():
            sys.stderr.write("ASyncP2PServer: start loop (%d sec interval).\n" % self.update_interval)
            self.scan()
        return
    
    def sendto(self, addr, data):
        if 2 <= self.debug:
            sys.stderr.write("send(%s): %s\n" % (addr, repr(data)))
        try:
            self.socket.sendto(self.signature+data, (addr, self.p2p_port))
        except socket.error:
            return
        return
    
    def handle_read(self):
        try:
            (data, (addr, port)) = self.socket.recvfrom(4096)
        except socket.error:
            return
        # ignore packets from outside, or ones whose signature is incorrect.
        if not inrange(addr, self.allow_ranges) or \
                     not data[:len(self.signature)] == self.signature:
            return
        if 2 <= self.debug:
            sys.stderr.write("recv(%s): %s\n" % (addr, repr(data)))
        data = data[len(self.signature):]
        if not data: return
        self.hostdb.intern_host(addr)
        if data[0] == "?":
            # handle address query
            self.sendto(addr, "!" + ' '.join(map(lambda host:host.addr,
                    filter(lambda host: host.alive, self.hostdb.allhosts()))))
        elif data[0] == "!":
            # receive address
            if data[1:]:
                for addr in data[1:].split(" "):
                    self.hostdb.intern_host(addr)
        elif data[0] == ">" and self.reporter:
            # handle query
            s = self.reporter.send_report(addr)
            self.sendto(addr, "<"+s)
        elif data[0] == "<":
            # receive status update
            if self.debug:
                sys.stderr.write("receive status from: %s\n" % addr)
            self.hostdb.receive_host_status(addr, data[1:])
        return
    
    def scan(self):
        # SIGALRM must not occur when updating the report.
        if self.reporter:
            self.reporter.update_report()
        self.hostdb.update_all_hosts()
        broadcast_ipquery = (self.tickcount % BROADQUERY_INTERVAL_COUNT) == 0
        if broadcast_ipquery and self.debug:
            sys.stderr.write("broadcast ipquery.\n")
        for host in self.hostdb.allhosts():
            if broadcast_ipquery:
                self.sendto(host.addr, "?")
            self.sendto(host.addr, ">")
        self.tickcount = self.tickcount + 1
        # reschedule myself again.
        signal.alarm(self.update_interval)
        return
    
##  ASyncHTTPServer
class ASyncHTTPServer(asyncore.dispatcher):
    def __init__(self, hostdb, allow_ranges, http_port, bindaddr="0.0.0.0"):
        asyncore.dispatcher.__init__(self)
        self.hostdb = hostdb
        self.create_socket(socket.AF_INET, socket.SOCK_STREAM)
        self.set_reuse_addr()
        self.http_port = http_port
        self.bind((bindaddr, http_port))
        self.listen(1)
        self.allow_ranges = allow_ranges
        sys.stderr.write("ASyncHTTPServer: allow: %s\n" % ', '.join(allow_ranges))
        sys.stderr.write("ASyncHTTPServer: listening: %s:%d\n" % (bindaddr, http_port))
        return
    
    def handle_accept(self):
        x = self.accept()
        if not x: return
        (conn, (addr,port)) = x
        if inrange(addr, self.allow_ranges):
            sys.stderr.write("ASyncHTTPServer: accepted: %s\n" % addr)
            ASyncHTTPService(conn, self.http_port, self.hostdb)
        else:
            sys.stderr.write("ASyncHTTPServer: refused: %s\n" % addr)
            conn.close()
        return

##
class ASyncHTTPService(asyncore.dispatcher):
    def __init__(self, sock, port, hostdb):
        asyncore.dispatcher.__init__(self, sock)
        self.phase = 0
        self.port = port
        self.hostdb = hostdb
        self.readbuf = ''
        self.writebuf = []
        self.writeidx = 0
        return
    
    def handle_read(self):
        s = self.recv(4096)
        if not s: return
        while '\n' in s:
            n = s.index('\n')
            (line, s) = (s[:n].strip(), s[n+1:])
            self.handle_line(self.readbuf+line)
            self.readbuf = ''
        self.readbuf = s
        return
    
    def handle_line(self, line):
        if line:
            f = line.split(' ')
            if f[0].lower() == 'get' and 1 < len(f):
                if f[1] == "/":
                    self.writebuf = [ "HTTP/1.0 200 ok\r\n"
                                        "Content-Type: text/html; charset=ISO-8859-1\r\n"
                                        "Pragma: no-cache\r\n"
                                        "Cache-Control: no-cache\r\n"
                                        "Expires: -1\r\n"
                                        "\r\n" ]
                    self.hostdb.make_html_report(self.writebuf, self.port)
                else:
                    self.writebuf = [ "HTTP/1.0 404 not found\r\n"
                                        "Content-Type: text/html; charset=ISO-8859-1\r\n"
                                        "\r\n<html><body>file does not exist</body></html>\n" ]
        else:
            self.phase = 1
        return
    
    def writable(self):
        return self.phase == 1 and self.writeidx < len(self.writebuf)
    
    def handle_write(self):
        assert self.phase == 1 and self.writeidx < len(self.writebuf)
        self.send(self.writebuf[self.writeidx])
        self.writeidx = self.writeidx + 1
        if len(self.writebuf) <= self.writeidx:
            self.handle_close()
        return
    
    def handle_close(self):
        self.close()
        return

# main routine
def main():
    import optparse
    parser = optparse.OptionParser()
    parser.remove_option('-h')
    parser.add_option('-d', '--debug', dest='debug', default=0, help='Debug level', action='store', type='int')
    parser.add_option('-u', '--update_interval', dest='update_interval', default=UPDATE_INTERVAL_SECONDS, help='Update interval(sec)', action='store', type='int')
    parser.add_option('-p', '--p2pport', dest='p2pport', default=P2P_PORT, help='P2P port', action='store', type='int')
    parser.add_option('-n', '--p2pallow', dest='p2pallow', help='P2P allow ranges', action='store', type='str')
    parser.add_option('-h', '--httpport', dest='httpport', default=HTTP_PORT, help='HTTP port', action='store', type='int')
    parser.add_option('-a', '--httpallow', dest='httpallow', help='HTTP allow ranges', action='store', type='str')
    parser.add_option('-s', '--scanaddr', dest='scanaddrs', help='P2P scan ranges', action='store', type='str')
    parser.add_option('-S', '--signature', dest='signature', default=SIGNATURE, help='P2P signature', action='store', type='str')
    parser.add_option('-H', '--nhistory', dest='nhistory', default=HISTORY_SIZE, help='History size', action='store', type='int')
    parser.add_option('-U', '--username', dest='username', default=UNAGI_USER, help='username', action='store', type='str')
    parser.add_option('-?', '--help', dest='help', help='show help message', action='store_true')

    def usage():
        parser.print_help()
        sys.exit(2)

    try:
        opts, args = parser.parse_args(sys.argv)
    except optparse.OptParseError:
        usage()

    if opts.help: usage()

    # fallback configurations
    if not opts.signature:
        opts.signature = 'localhost'
    if not opts.username:
        import pwd
        pw_struct = pwd.getpwuid(os.getuid())
        opts.username = pw_struct.pw_name

    if not opts.signature:
        sys.stderr.write("Set your network SIGNATURE!\n")
        sys.exit(1)
    if not opts.p2pport:
        sys.stderr.write("Set P2P_PORT!\n")
        sys.exit(1)
    if not opts.httpport:
        sys.stderr.write("Set HTTP_PORT!\n")
        sys.exit(1)

    if not opts.scanaddrs:
        opts.scanaddrs = opts.p2pallow or P2P_SCAN_RANGES
    if not opts.p2pallow:
        opts.p2pallow = P2P_ALLOW_RANGES
    if not opts.httpallow:
        opts.httpallow = HTTP_ALLOW_RANGES
    if opts.username:
        import pwd
        pw_struct = pwd.getpwnam(opts.username)
        os.setgid(pw_struct.pw_gid)
        # only root can os.setgroups([])
        if hasattr(os, 'setgroups') and pw_struct.pw_uid == 0:
            # only available in Python 2.0 or later
            os.setgroups([])
        os.setuid(pw_struct.pw_uid)
        
    hostdb = HostDB(opts.scanaddrs, opts.nhistory, opts.update_interval)
    reporter = LinuxStatusReporter()
    ASyncP2PServer(opts.update_interval, hostdb, reporter, opts.signature, opts.p2pallow, opts.p2pport, debug=opts.debug)
    ASyncHTTPServer(hostdb, opts.httpallow, opts.httpport)

    while True:
        try:
            asyncore.loop()
        except select.error, err:
            if err[0] != errno.EINTR:
                raise
    
if __name__ == "__main__":
    main()
