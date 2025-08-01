from nptdms import TdmsFile

tdms_file = TdmsFile.read("testFiles/VE_UTC_20240517_155411.892.tdms")

print("Available groups and channels:")
for group in tdms_file.groups():
    print(f"Group: {group.name}")
    for channel in group.channels():
        print(f"  Channel: {channel.name}, Length: {len(channel[:])}")
