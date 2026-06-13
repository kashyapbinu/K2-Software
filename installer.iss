; Inno Setup script for K2 Aerospace
; Build the app first:   pyinstaller K2.spec --noconfirm
; Then compile this:     ISCC.exe installer.iss   (Inno Setup 6+)
; Produces:              installer_output\K2-Setup.exe
;
; Installs the one-dir PyInstaller bundle into Program Files, adds Start-menu
; and (optional) desktop shortcuts, and registers a proper uninstaller — so the
; user gets a normal "installed application that opens as an app" experience.

#define MyAppName "K2 Aerospace"
#define MyAppVersion "0.1.0"
#define MyAppPublisher "Kashyap Binu"
#define MyAppURL "https://github.com/kashyapbinu/K2-Software"
#define MyAppExeName "K2.exe"

[Setup]
AppId={{8F3C2A91-K2AE-4B7D-9E21-K2AEROSIM0001}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
; Per-machine install needs admin; switch to lowest + {localappdata} for per-user.
PrivilegesRequired=admin
OutputDir=installer_output
OutputBaseFilename=K2-Setup
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
; Uncomment once bin\k2.ico exists:
; SetupIconFile=bin\k2.ico

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional icons:"

[Files]
; Pull in the entire one-dir bundle produced by PyInstaller.
Source: "dist\K2\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent
