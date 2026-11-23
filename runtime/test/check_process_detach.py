"""Native Windows oracle for synthetic mapped EXE/DLL termination; no game/GPU dependencies."""
import argparse
import json
import os
from pathlib import Path
import struct
import subprocess


def run(command, mode, scenario, evidence):
    env = os.environ.copy()
    env.update(LEONIDA_DETACH_MODE=mode, LEONIDA_DETACH_CASE=scenario)
    result = subprocess.run(command, env=env, capture_output=True, timeout=10)
    assert len(result.stdout) % 24 == 0, repr(result.stdout)
    events = list(struct.iter_unpack('<6I', result.stdout))
    assert all(e[0] == 0x52543337 for e in events), events
    if evidence:
        evidence.mkdir(parents=True, exist_ok=True)
        variant = 'native' if len(command) == 1 else 'early-control' if len(command) == 3 else 'mapped'
        name = variant + '-' + scenario + '-' + mode
        (evidence / (name + '.json')).write_text(json.dumps({
            'command': command, 'exit': result.returncode, 'events': events,
            'stderr': result.stderr.decode('utf-8', 'replace')}, indent=2), encoding='utf-8', newline='\n')
    code = 42 if scenario == 'ExitThread' else 3 if mode == 'abort' else 23
    assert result.returncode == code, (command, result.returncode, result.stderr)
    return events


def check(events, mode, scenario):
    detaches = [e for e in events if e[2] in (0, 10)]
    if mode == 'TerminateProcess' or scenario == 'map_only':
        assert not detaches, detaches
        return []
    # ExitThread leaves no living process-exit caller. Check status/no hang only, not TLS parity on the host.
    if scenario == 'ExitThread':
        return []
    assert [e[4] for e in events if e[1:3] == (1, 100)] == [0x6A], events
    expected = [(2, 0), (2, 10), (1, 0)]
    if scenario == 'reentry':
        # A recursive ExitProcess can stop the remaining notifications; use the native result as the oracle.
        assert detaches and detaches[0][1:3] == (2, 0), detaches
    else:
        assert [e[1:3] for e in detaches] == expected, detaches
    tid = detaches[0][3]
    assert all(e[3] == tid for e in detaches), detaches
    last_exe = [e[4] for e in events if e[1:4] == (1, 101, tid)][-1]
    last_dll = [e[4] for e in events if e[1:4] == (2, 100, tid)][-1]
    for e in detaches:
        assert e[4] == (last_exe if e[1] == 1 else last_dll), (e, last_exe, last_dll)
        assert e[5] == (1 if e[2] == 10 else 0), e
    if scenario in ('worker_exit', 'entry_with_worker'):
        assert len([e for e in events if e[2] == 200]) == 1, events
        assert not [e for e in events if e[2] == 3], events  # ExitProcess does not issue thread detach
        if scenario == 'entry_with_worker':
            before = [e[4] for e in events if e[2] == 201]
            after = [e[4] for e in events if e[2] == 202]
            assert len(before) == 1 and before == after, (before, after)  # worker cannot execute during detach
    return [(e[1], e[2], e[4], e[5]) for e in detaches]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--driver', required=True)
    parser.add_argument('--guest', required=True)
    parser.add_argument('--evidence', type=Path)
    args = parser.parse_args()
    modes = ('return', 'exit', '_exit', '_Exit', 'quick_exit', 'ExitProcess', 'TerminateProcess', 'abort')
    cases = [(s, m) for s in ('entry', 'entry_with_worker', 'worker_exit') for m in modes]
    cases += [('reentry', 'ExitProcess')]
    cases += [('racing_exit', 'ExitProcess')] * 20
    count = 0
    for scenario, mode in cases:
        native = check(run([args.guest], mode, scenario, args.evidence), mode, scenario)
        mapped = check(run([args.driver, args.guest], mode, scenario, args.evidence), mode, scenario)
        # Race winners can have different TLS values in different processes; each is checked against its own log.
        if scenario != 'racing_exit':
            assert mapped == native, (scenario, mode, native, mapped)
        count += 2
        print(f'PASS {scenario}/{mode}', flush=True)
    check(run([args.driver, args.guest], 'return', 'map_only', args.evidence), 'return', 'map_only')
    check(run([args.driver, args.guest], 'return', 'ExitThread', args.evidence), 'return', 'ExitThread')
    early = run([args.driver, args.guest, '--early-detach-control'], 'ExitProcess', 'entry_with_worker', args.evidence)
    before = [e[4] for e in early if e[2] == 201]
    after = [e[4] for e in early if e[2] == 202]
    assert len(before) == len(after) == 1 and after[0] > before[0], (before, after)
    print(f'PASS: {count + 3} child processes, including 20 independent exit races per implementation; '
          'early-detach control confirms the worker can run before native shutdown')


if __name__ == '__main__':
    main()
