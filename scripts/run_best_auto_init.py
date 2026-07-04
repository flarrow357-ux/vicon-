from pathlib import Path

from viconnexusapi import ViconNexus


PIPELINE_PATH = Path(
    r"C:\Users\Public\Documents\Vicon\Nexus2.x\Configurations\Pipelines\temp_AutoInitializeProbe.Pipeline"
)
FRAME = 200

PIPELINE_PATH.write_text(
    f"""<?xml version="1.1" encoding="UTF-8" standalone="no" ?>
<Pipeline>
  <Entry DisplayName="Autolabel Static Frame" Enabled="1" OperationId="25" OperationName="TPoseLabel">
    <ParamList name="" version="1">
      <Param name="FRAME" value="{FRAME}"/>
      <Param name="Tolerance" value="0.025"/>
      <Param name="SeparationDistance" value="1"/>
      <Param name="ClearLabels" value="1"/>
      <Param macro="ACTIVE_SUBJECTS" name="SUBJECTS"/>
    </ParamList>
  </Entry>
</Pipeline>
""",
    encoding="utf-8",
)

nexus = ViconNexus.ViconNexus()
nexus.RunPipeline("temp_AutoInitializeProbe", "Shared", 120)
print("unlabeled", nexus.GetUnlabeledCount())
nexus.SaveTrial(120)
print("saved", nexus.GetTrialName())
