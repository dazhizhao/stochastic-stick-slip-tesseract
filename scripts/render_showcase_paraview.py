"""Render one showcase trajectory with ParaView using locked visual scales."""

import argparse
import json
from pathlib import Path

from paraview.simple import (
    ColorBy,
    CreateView,
    GetColorTransferFunction,
    GetScalarBar,
    PVDReader,
    Render,
    SaveScreenshot,
    SetActiveView,
    Show,
    Sphere,
    Text,
    WarpByVector,
)


STICK_COLOR = [0.08, 0.20, 0.38]
SLIP_COLOR = [0.92, 0.42, 0.12]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--series", choices=("fixed", "trained"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    metadata = json.loads(args.metadata.read_text())
    series = metadata[args.series]
    args.output.mkdir(parents=True, exist_ok=True)

    reader = PVDReader(
        registrationName=args.series,
        FileName=str(Path(series["pvd"]).resolve()),
    )
    warp = WarpByVector(registrationName="deformed", Input=reader)
    warp.Vectors = ["POINTS", "displacement"]
    warp.ScaleFactor = metadata["deformation_scale"]

    view = CreateView("RenderView")
    SetActiveView(view)
    view.ViewSize = [900, 360]
    view.UseColorPaletteForBackground = 0
    view.Background = [1.0, 1.0, 1.0]
    view.Background2 = [1.0, 1.0, 1.0]
    view.OrientationAxesVisibility = 0
    view.CameraParallelProjection = 1
    view.CameraPosition = [0.5, 0.05, 2.0]
    view.CameraFocalPoint = [0.5, 0.05, 0.0]
    view.CameraViewUp = [0.0, 1.0, 0.0]
    view.CameraParallelScale = metadata["camera_parallel_scale"]

    display = Show(warp, view, "UnstructuredGridRepresentation")
    display.Representation = "Surface With Edges"
    display.EdgeColor = [0.20, 0.22, 0.25]
    display.LineWidth = 0.7
    ColorBy(display, ("POINTS", "displacement_magnitude"))
    color_map = GetColorTransferFunction("displacement_magnitude")
    displacement_max = metadata["displacement_max"]
    color_map.RGBPoints = [
        0.0,
        0.267,
        0.005,
        0.329,
        0.33 * displacement_max,
        0.191,
        0.407,
        0.556,
        0.67 * displacement_max,
        0.208,
        0.719,
        0.473,
        displacement_max,
        0.993,
        0.906,
        0.144,
    ]
    color_map.ColorSpace = "Lab"
    color_map.RescaleTransferFunction(0.0, metadata["displacement_max"])
    display.SetScalarBarVisibility(view, True)
    scalar_bar = GetScalarBar(color_map, view)
    scalar_bar.Title = "Displacement"
    scalar_bar.ComponentTitle = "magnitude"
    scalar_bar.TitleColor = [0.08, 0.08, 0.08]
    scalar_bar.LabelColor = [0.08, 0.08, 0.08]
    scalar_bar.TitleFontSize = 16
    scalar_bar.LabelFontSize = 14
    scalar_bar.WindowLocation = "Any Location"
    scalar_bar.Position = [0.84, 0.17]
    scalar_bar.ScalarBarLength = 0.60

    marker_sources = []
    marker_displays = []
    for index in range(2):
        marker = Sphere(registrationName=f"contact_{index}")
        marker.Radius = 0.014
        marker.ThetaResolution = 24
        marker.PhiResolution = 24
        marker_sources.append(marker)
        marker_display = Show(marker, view, "GeometryRepresentation")
        marker_display.AmbientColor = STICK_COLOR
        marker_display.DiffuseColor = STICK_COLOR
        marker_displays.append(marker_display)

    label = Text(registrationName="series_label")
    label.Text = series["label"]
    label_display = Show(label, view, "TextSourceRepresentation")
    label_display.WindowLocation = "Upper Left Corner"
    label_display.FontSize = 20
    label_display.Color = [0.05, 0.05, 0.05]

    time_label = Text(registrationName="time_label")
    time_display = Show(time_label, view, "TextSourceRepresentation")
    time_display.WindowLocation = "Upper Right Corner"
    time_display.FontSize = 16
    time_display.Color = [0.05, 0.05, 0.05]

    for frame, time_value in enumerate(metadata["times"]):
        view.ViewTime = time_value
        reader.UpdatePipeline(time_value)
        warp.UpdatePipeline(time_value)
        view.CameraPosition = [0.5, 0.05, 2.0]
        view.CameraFocalPoint = [0.5, 0.05, 0.0]
        view.CameraViewUp = [0.0, 1.0, 0.0]
        view.CameraParallelProjection = 1
        view.CameraParallelScale = metadata["camera_parallel_scale"]
        time_label.Text = f"t = {time_value:.3f}"
        for contact in range(2):
            marker_sources[contact].Center = series["contact_centers"][frame][contact]
            color = (
                SLIP_COLOR
                if series["contact_state"][frame][contact]
                else STICK_COLOR
            )
            marker_displays[contact].AmbientColor = color
            marker_displays[contact].DiffuseColor = color
        Render(view)
        SaveScreenshot(
            str(args.output / f"{args.series}_{frame:04d}.png"),
            view,
            ImageResolution=[900, 360],
            TransparentBackground=0,
        )


if __name__ == "__main__":
    main()
