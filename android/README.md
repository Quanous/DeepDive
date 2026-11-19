# HL for Android

`DL/DL.apk` is the same single-file planner (`DL/DL.html`) wrapped in a tiny WebView app (`com.heeseop.hl`).
It works offline, needs no permissions, and keeps its data inside the app.

Rebuild: `python3 android/build_apk.py <tools-dir> <work-dir>` (tools: apktool-cli for aapt2 + framework resources,
android-all for compiling, dalvik-dx, apksig - all from Maven Central).
The signing key (`android/hl.p12`) is intentionally NOT in the repository. Updates only install over an existing
install when they are signed with the same key; otherwise uninstall first (back up your data with the Backup button).
