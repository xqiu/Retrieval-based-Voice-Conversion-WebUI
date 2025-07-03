#no parameter, use UI to enter
#1. source *.wav file
#2. output *.wav file
#3. index model file (*.index select from same folder as this script)
#4. index rate (0.6-0.75)
#finally, generate command: python tools\infer_cli.py --f0up_key 0 --input_path {source.wav} --opt_path {output.wav} --index_path {model.index} --f0method rmvpe --model_name "jnbodhi3.pth" --index_rate {rate} --device cuda:0 --is_half True

#"chapter" of this script (enclosed by BEGIN and END)
#1. generate WPF UI
#2. constant
#3. helper function
#4. Get all the controls
#5. bind event

#BEGIN generate WPF UI
Add-Type -AssemblyName PresentationFramework
Add-Type -AssemblyName PresentationCore
Add-Type -AssemblyName WindowsBase

# Create the XAML
# first line: label "Source", textbox for source file, and button "Browse" (open file dialog limit to *.wav)
# second line: label "Output", textbox for output file, and button "Browse" (save file dialog limit to *.wav)
# third line: label "Index", textbox for index file, and button "Browse" (open file dialog limit to *.index)
# fourth line: label "Rate" and number for index rate (0.6-0.75)
# fifth line: button "Run"
$xaml = @"
<Window xmlns="http://schemas.microsoft.com/winfx/2006/xaml/presentation" 
        xmlns:x="http://schemas.microsoft.com/winfx/2006/xaml"
        Title="Infer CLI" Height="200" Width="480">
    <Grid>
        <Grid.RowDefinitions>
            <RowDefinition Height="Auto"/>
            <RowDefinition Height="Auto"/>
            <RowDefinition Height="Auto"/>
            <RowDefinition Height="Auto"/>
            <RowDefinition Height="Auto"/>
            <RowDefinition Height="Auto"/>
        </Grid.RowDefinitions>
        <Grid.ColumnDefinitions>
            <ColumnDefinition Width="Auto"/>
            <ColumnDefinition Width="*"/>
            <ColumnDefinition Width="Auto"/>
        </Grid.ColumnDefinitions>

        <Label Grid.Row="0" Grid.Column="0" Content="Source"/>
        <TextBox Grid.Row="0" Grid.Column="1" Name="SourceTextBox"/>
        <Button Grid.Row="0" Grid.Column="2" Content="Browse" Name="SourceBrowseButton"/>

        <Label Grid.Row="1" Grid.Column="0" Content="Output"/>
        <TextBox Grid.Row="1" Grid.Column="1" Name="OutputTextBox"/>
        <Button Grid.Row="1" Grid.Column="2" Content="Browse" Name="OutputBrowseButton"/>

        <Label Grid.Row="2" Grid.Column="0" Content="Model"/>
        <ComboBox Grid.Row="2" Grid.Column="1" Name="ModelComboBox"/>

        <Label Grid.Row="3" Grid.Column="0" Content="Index"/>
        <TextBox Grid.Row="3" Grid.Column="1" Name="IndexTextBox"/>
        <Button Grid.Row="3" Grid.Column="2" Content="Browse" Name="IndexBrowseButton"/>

        <Button Grid.Row="5" Grid.Column="1" Content="Run" Name="RunButton"/>
    </Grid>
</Window>
"@
#END generate WPF UI

# Load Forms for folder dialogs
Add-Type -AssemblyName System.Windows.Forms

#BEGIN constant
$thisScriptPath = $MyInvocation.MyCommand.Path
$thisScriptDir = Split-Path -Path $thisScriptPath -Parent
## Determine Python executable path
$pythonExePath = "$pwd\venv_fastinstall\python.exe"
# If default venv path doesn't exist, try 'venv/Scripts/python.exe'
if (-not (Test-Path $pythonExePath)) {
    $altPy = "$pwd\venv\Scripts\python.exe"
    if (Test-Path $altPy) {
        $pythonExePath = $altPy
    } else {
        # Fallback to system python
        $pythonExePath = "python"
    }
}

$inputFileName = "run_infer_preference.xml"
$logFileName = "run_infer.log"
#END constant

#BEGIN helper function
# log message
function LogMessage {
  param (
    [string]$message,
    [ValidateSet("INFO", "WARNING", "ERROR")]
    [string]$level = "INFO"
  )
  #prepend level timestamp
  $message = "[$level]`n`t$(Get-Date -Format "yyyy-MM-dd HH:mm:ss")`n`t$message"
  $message | Out-File -FilePath $logFileName -Append
}

# save my input value
function SaveInput {
  param (
    [string]$source,
    [string]$output,
    [string]$model,
    [string]$index
  )
  $myinput = @{
    source = $source
    output = $output
    model  = $model
    index  = $index
  }
  $myinput | Export-Clixml -Path $inputFileName
}
# load my input value
function LoadInput {
  $myinput = if (Test-Path $inputFileName) { Import-Clixml -Path $inputFileName } else { @{} }

  #set value if exist
  $SourceTextBox.Text = if($null -ne $myinput.source) { $myinput.source } else { "" }
  $OutputTextBox.Text = if($null -ne $myinput.output) { $myinput.output } else { "" }
  $IndexTextBox.Text = if($null -ne $myinput.index) { $myinput.index } else { "" }
  $ModelComboBox.SelectedItem = if($null -ne $myinput.model) { $myinput.model } else { "" }

  if($ModelComboBox.SelectedIndex -eq -1) {
    $ModelComboBox.SelectedIndex = 0
  }
}

function GetModelList {
  $modelDir = "$thisScriptDir\assets\weights"
  $modelFiles = Get-ChildItem -Path $modelDir -Filter "*.pth"
  $modelFiles | ForEach-Object {
    $ModelComboBox.Items.Add($_.Name)
  }
}

function CheckAndInstallFFMPEG {
  cmd /c "where ffmpeg"
  if(0 -eq $LastExitCode) {
    return
  }

  #download ffmpeg from https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip as ffmpeg.zip
  wget -OutFile ffmpeg.zip https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip
  Expand-Archive -Path ffmpeg.zip -DestinationPath .
  Remove-Item ffmpeg.zip
  #move ffmpeg to $thisScriptDir
  Move-Item -Path .\ffmpeg-7.1-essentials_build\bin\ffmpeg.exe -Destination $thisScriptDir
  Move-Item -Path .\ffmpeg-7.1-essentials_build\bin\ffprobe.exe -Destination $thisScriptDir
}

function RunCmdFile {
  param (
    #plase use fullpath
    [string]$cmdFile,
    [string]$workingDir = $thisScriptDir
  )
  
  Start-Process $cmdFile -WorkingDirectory:$workingDir -NoNewWindow -Wait
}
#END helper function

# Load the XAML
[xml]$xaml = $xaml
$reader = (New-Object System.Xml.XmlNodeReader $xaml)
$window = [Windows.Markup.XamlReader]::Load($reader)
# BEGIN Get all the controls
$SourceTextBox = $window.FindName("SourceTextBox")
$SourceBrowseButton = $window.FindName("SourceBrowseButton")

$OutputTextBox = $window.FindName("OutputTextBox")
$OutputBrowseButton = $window.FindName("OutputBrowseButton")

$ModelComboBox = $window.FindName("ModelComboBox")

$IndexTextBox = $window.FindName("IndexTextBox")
$IndexBrowseButton = $window.FindName("IndexBrowseButton")

$RunButton = $window.FindName("RunButton")
#END Get all the controls

#BEGIN bind event

# Source Browse Button
$SourceBrowseButton.Add_Click({
    # Select source directory
    $folderDialog = New-Object System.Windows.Forms.FolderBrowserDialog
    $folderDialog.Description = "Select Source Directory"
    if ($folderDialog.ShowDialog() -eq 'OK') {
        $SourceTextBox.Text = $folderDialog.SelectedPath
    }
  })

# Output Browse Button
$OutputBrowseButton.Add_Click({
    # Select output directory
    $folderDialog = New-Object System.Windows.Forms.FolderBrowserDialog
    $folderDialog.Description = "Select Output Directory"
    $folderDialog.ShowNewFolderButton = $true
    if ($folderDialog.ShowDialog() -eq 'OK') {
        $OutputTextBox.Text = $folderDialog.SelectedPath
    }
  })

# Index Browse Button
$IndexBrowseButton.Add_Click({
    $openFileDialog = New-Object Microsoft.Win32.OpenFileDialog
    $openFileDialog.Filter = "Index Files (*.index)|*.index"
    $openFileDialog.Title = "Select a Index File"
    $openFileDialog.ShowDialog() | Out-Null
    $IndexTextBox.Text = $openFileDialog.FileName
  })

# Run Button

$RunButton.Add_Click({
    $source = $SourceTextBox.Text
    $output = $OutputTextBox.Text
    $index = $IndexTextBox.Text
    $model = $ModelComboBox.SelectedItem

    #WARNING: $model and $index must be from same training process, but we has no way to check it

    #show warning dialog if source, output, index is empty
    if ($source -eq "" -or $output -eq "") {
      [System.Windows.MessageBox]::Show("Source, Output is required", "Warning", [System.Windows.MessageBoxButton]::OK)
      return
    }

    #BEGIN command
    $command = @(
      "run_infer.py",
      "--input_dir", "$source",
      "--output_dir", "$output",
      "--model", "$model"
    )
    #only add index_path if index is not empty and that file existed
    $hasIndexFile = $index -ne "" -and (Test-Path $index)
    if ($hasIndexFile) {
      $command += "--index_path"
      $command += $index
    }

    $commandStr = $command -join " "
    #END command
  
    #Show dialog with button OK and Cancel
    $indexFileText = if ($hasIndexFile) { "Index File: $index" } else { "No Index File" }
    $result = [System.Windows.MessageBox]::Show("Will Run command with`n`nSource: $source`n`nOutput: $output`n`n$indexFileText`n`n=> $commandStr", 
      "Command", [System.Windows.MessageBoxButton]::OKCancel)
    if ($result -eq "Cancel") {
      return
    }

    #there is no good way to run command asynchronously in PowerShell(or cause crash), so run synchronously
    #and progress bar is not working(due to ui thread is blocked)
    
    #log command
    LogMessage "Command: $commandStr"
    $commandOutput = & $pythonExePath $command 2>&1 | Out-String
    #if successfully exit, $LastExitCode should be 0
    if ($LastExitCode -ne 0) {
      LogMessage $commandOutput -level "ERROR"
      [System.Windows.MessageBox]::Show($commandOutput, "Command Output", [System.Windows.MessageBoxButton]::OK)
    }
    else {
      LogMessage $commandOutput
      [System.Windows.MessageBox]::Show("Command successfully executed.`n FILE in => $output", "Command Output", [System.Windows.MessageBoxButton]::OK)
      #open dir of output file
      Invoke-Item $output
    }
  })

# on close, save my input value
$window.Add_Closing({
    SaveInput -source:$SourceTextBox.Text -output:$OutputTextBox.Text -index:$IndexTextBox.Text -rate:$RateTextBox.Text -model:$ModelComboBox.SelectedItem
  })
#only run command if first time show
$window.Add_IsVisibleChanged({
  if ($window.IsVisible) {
    #check neccessary data/tool before run gui
    CheckAndInstallFFMPEG

    #run some *.bat file if needed
    #RunCmdFile

    # Load model list
    GetModelList
    # Load my input value
    LoadInput
  }
})
#END bind event

# Show the window
$window.ShowDialog() | Out-Null
