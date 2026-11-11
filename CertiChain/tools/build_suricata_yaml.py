"""Build data/rules/suricata.yaml from the Homebrew default by targeted text edits (keeps comments)."""
import glob, os, sys
SRC = "/opt/homebrew/etc/suricata/suricata.yaml"
RDIR = str(__import__("pathlib").Path(__file__).resolve().parents[1] / "data" / "rules")
RULES = f"{RDIR}/et-open-suricata-8.0/rules"
OUT = f"{RDIR}/suricata.yaml"
LOGDIR = f"{RDIR}/logs/"
t = open(SRC).read()

def sub(old, new, count=1):
    global t
    n = t.count(old)
    if n != count:
        sys.exit(f"expected {count} occurrence(s) of {old!r}, found {n}")
    t = t.replace(old, new)

header = ("# IoTJ-SI26-Oct project config: copy of /opt/homebrew/etc/suricata/suricata.yaml (Suricata 8.0.7)\n"
          "# edited by build_suricata_yaml.py on 2026-10-18. Changes vs default: EXTERNAL_NET=any, default-log-dir,\n"
          "# eve-log (pcap-file, community-id, ethernet, alert payload/packet/metadata/rule raw, verdict),\n"
          "# TLS ja3/ja4 fingerprints on, rule path -> ET Open (suricata-8.0 tarball), local classification/reference/threshold.\n"
          "# Override the output dir per run with:  -l <outdir>   (or --set default-log-dir=<outdir>)\n")
sub("%YAML 1.1\n---\n", "%YAML 1.1\n---\n" + header)
sub('    EXTERNAL_NET: "!$HOME_NET"\n    #EXTERNAL_NET: "any"\n',
    '    #EXTERNAL_NET: "!$HOME_NET"   # Homebrew default\n'
    '    # Offline lab-dataset analysis: attackers and victims are usually both in RFC1918 space,\n'
    '    # so "$EXTERNAL_NET -> $HOME_NET" rules would never fire with the default. Use "any".\n'
    '    EXTERNAL_NET: "any"\n')
sub("default-log-dir: /opt/homebrew/var/log/suricata/\n", f"default-log-dir: {LOGDIR}\n")
sub("      pcap-file: false\n", "      pcap-file: true\n")
sub("      community-id: false\n", "      community-id: true\n")
sub("      #ethernet: no  # log ethernet header in events when available\n",
    "      ethernet: yes  # log ethernet header (src/dest MAC) in events when available\n")
old_alert = t[t.index("        - alert:\n            # payload: yes"):t.index("        # app layer frames\n")]
new_alert = """        - alert:
            payload: yes             # payload in Base64
            payload-buffer-size: 4 KiB
            payload-printable: yes   # payload in printable (lossy) format
            payload-length: yes
            packet: yes              # dump of the triggering packet (Base64, without stream segments)
            metadata:
              app-layer: true        # decoded app-layer (http, dns, tls, ...) attached to the alert
              flow: true             # flow record state (pkts/bytes/start) attached to the alert
              rule:
                metadata: true       # rule metadata (e.g. mitre_tactic_id / mitre_technique_id, signature_severity)
                raw: true            # raw rule text
                reference: true      # rule references (CVE, URLs)
            tagged-packets: yes
            verdict: yes
"""
t = t.replace(old_alert, new_alert)
sub("      #ja3-fingerprints: auto\n", "      ja3-fingerprints: yes\n")
sub("      #ja4-fingerprints: auto\n", "      ja4-fingerprints: yes\n")
files = sorted(os.path.basename(p) for p in glob.glob(f"{RULES}/*.rules"))
rf = "rule-files:\n" + "".join(f"  - {f}\n" for f in files)
sub("default-rule-path: /opt/homebrew/var/lib/suricata/rules\n\nrule-files:\n  - suricata.rules\n",
    f"default-rule-path: {RULES}\n\n# all {len(files)} *.rules files of ET Open (same set suricata-update would merge); commented-out rules stay disabled\n" + rf)
sub("classification-file: /opt/homebrew/etc/suricata/classification.config\n",
    f"classification-file: {RULES}/classification.config\n")
sub("reference-config-file: /opt/homebrew/etc/suricata/reference.config\n",
    f"reference-config-file: {RDIR}/conf/reference.config\n")
sub("# threshold-file: /opt/homebrew/etc/suricata/threshold.config\n",
    f"threshold-file: {RDIR}/conf/threshold.config\n")
open(OUT, "w").write(t)
print("wrote", OUT, "rule files:", len(files))
