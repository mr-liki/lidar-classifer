NAKSHA -- LAS/LAZ 1.4 to compatible LAZ 1.2 batch converter
==========================================================
INPUT (configured): \\192.168.10.6\d\NT895\A13\LAS
OUTPUT (configured): \\192.168.10.6\d\NT895\A13\LAS\LAZ_1_2

WHAT TO DO:
1. Download free LASzip (laszip64.exe or laszip.exe) from https://rapidlasso.de/downloads/ or use your existing LAStools bin folder.
2. Put LASzip executable beside convert_las14_to_laz12.py OR set LASZIP_EXE path in RUN_BATCH_LAS14_TO_LAZ12.bat.
3. Make sure the Windows machine can access the UNC folder (Explorer: \\192.168.10.6\d\NT895\A13\LAS).
4. Double-click SCAN_FIRST_NO_CONVERSION.bat; it reads the headers without changing any data.
5. Double-click RUN_BATCH_LAS14_TO_LAZ12.bat to convert every supported tile.
6. See conversion_report_*.csv in LAZ_1_2 and _conversion_logs for errors.

CAUTIONS:
- LASzip compatibility mode is required for 1.4 point formats 6, 7, 8 to obtain 1.2 legacy-format LAZ files. Default native LASzip compression is NOT sufficient.
- Format 6 -> 1; formats 7 and 8 -> 3; class codes above 31 / extended flags / return data are embedded as compatibility Extra Bytes. Older 1.2 consumers may not expose them, even though LASzip can reconstruct them.
- Point formats 9 and 10 (waveforms) cannot be LAS 1.2 and are rejected. Point format 0..3 LAS1.4 files require an independent controlled header downgrade path and are deliberately not handled by this converter.
- LAS1.2 supports <=4,294,967,295 points in a file.
- Original inputs are never altered and existing output files are NEVER overwritten. Interrupted conversions use temporary .partial.laz files and are not installed as final output.
- A header and VLR check is performed after every conversion, but this does not replace a full decompress-and-compare QA run on representative files.
- Conversion is sequential for dependable network-share operation; no extra Python packages are needed.

PILOT (first file):
  python convert_las14_to_laz12.py --limit 1 --laszip "C:\LAStools\bin\laszip64.exe"

RUN FULL BATCH:
  python convert_las14_to_laz12.py --laszip "C:\LAStools\bin\laszip64.exe"

ONLY HEADER AUDIT:
  python convert_las14_to_laz12.py --scan-only

FIRST FILE + FULL READ-BACK CHECK:
  Double-click RUN_FIRST_FILE_PILOT.bat
  Or: python convert_las14_to_laz12.py --limit 1 --deep-verify

OPTIONAL WHOLE-BATCH DEEP INTEGRITY SCAN (SLOWER):
  python convert_las14_to_laz12.py --deep-verify
