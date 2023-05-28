# -*- coding: utf-8 -*-
"""
Export Colour Dataset
=====================

Unlike *Colour - Ocean*, only exports a cherry-picked subset of *Colour*
datasets for *Mitsuba 2* renderer.
"""

import numpy as np
import os
import re
import xml.dom.minidom
from scipy.optimize import fmin

import colour
import colour_datasets
import xml.etree.ElementTree as ET

__author__ = "Colour Developers"
__copyright__ = "Copyright (C) 2019-2021 - Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "MITSUBA_SHAPE",
    "slugify",
    "format_spectrum",
    "scale_sd_to_luminous_flux",
    "export_AMPAS_training_data_bsdfs_files",
    "export_colorchecker_classic_bsdfs_files",
    "export_colorchecker_classic_support_bsdfs_file",
    "export_emitters_files",
    "export_synthetic_LEDs",
    "export_emitters_bt2020",
]

MITSUBA_SHAPE = colour.SpectralShape(360, 830, 5)


def slugify(a):
    return re.sub(
        r"\s|-|\.", "_", re.sub(r"(?u)[^-\w.]", " ", str(a).strip()).strip().lower()
    )


def format_spectrum(sd, decimals=15):
    return ", ".join(
        "{1}:{2:0.{0}f}".format(decimals, wavelength, max(0, value))
        for wavelength, value in zip(sd.wavelengths, sd.values)
    )


def scale_sd_to_luminous_flux(sd, luminous_flux):
    base_sd = sd.copy()

    def function_error(scale, luminous_flux):
        sd = base_sd * scale
        error = np.linalg.norm(colour.luminous_flux(sd) - luminous_flux)

        return error

    scale = fmin(function_error, 1, (luminous_flux,), **{"disp": False})

    return base_sd * scale


def export_AMPAS_training_data_bsdfs_files(
    bsdf_type="diffuse", ior=1.460, alpha=0.05, output_directory="include"
):
    sensitivities_database = colour_datasets.load(
        "RAW to ACES Utility Data - Dyer et al. (2017)"
    )

    training_data = sensitivities_database["training"]["190-patch"]

    scene = ET.Element("scene", attrib={"version": "2.0.0"})
    for sd in colour.colorimetry.sds_and_msds_to_sds(training_data):
        bsdf = ET.SubElement(
            scene,
            "bsdf",
            attrib={"type": bsdf_type, "id": sd.name.split("-")[0].strip()},
        )

        ET.SubElement(
            bsdf,
            "spectrum",
            attrib={
                "name": (
                    "reflectance" if bsdf_type == "diffuse" else "diffuse_reflectance"
                ),
                "value": format_spectrum(sd),
            },
        )

        if bsdf_type in ("plastic", "roughplastic"):
            ET.SubElement(bsdf, "float", attrib={"name": "int_ior", "value": str(ior)})

        if bsdf_type == "roughplastic":
            ET.SubElement(bsdf, "float", attrib={"name": "alpha", "value": str(alpha)})

    with open(
        os.path.join(output_directory, "bsdfs_190_patch_{}.xml".format(bsdf_type)), "w"
    ) as xml_file:
        xml_file.write(
            xml.dom.minidom.parseString(ET.tostring(scene)).toprettyxml(indent=" " * 4)
        )


def export_colorchecker_classic_bsdfs_files(
    colour_checker="BabelColor Average", output_directory="colorchecker_classic/include"
):
    scene = ET.Element("scene", attrib={"version": "2.0.0"})
    for sd in colour.SDS_COLOURCHECKERS[colour_checker].values():
        bsdf = ET.SubElement(
            scene,
            "bsdf",
            attrib={
                "type": "diffuse",
                "id": "colorchecker_classic_{0}".format(slugify(sd.name)),
            },
        )

        ET.SubElement(
            bsdf,
            "spectrum",
            attrib={"name": "reflectance", "value": format_spectrum(sd)},
        )

    with open(
        os.path.join(output_directory, "bsdfs_{0}.xml".format(slugify(colour_checker))),
        "w",
    ) as xml_file:
        xml_file.write(
            xml.dom.minidom.parseString(ET.tostring(scene)).toprettyxml(indent=" " * 4)
        )


def export_colorchecker_classic_support_bsdfs_file(
    output_directory="colorchecker_classic/include",
):
    scene = ET.Element("scene", attrib={"version": "2.0.0"})
    bsdf = ET.SubElement(
        scene,
        "bsdf",
        attrib={"type": "roughplastic", "id": "colorchecker_classic_support"},
    )

    ET.SubElement(
        bsdf,
        "spectrum",
        attrib={
            "name": "diffuse_reflectance",
            "value": format_spectrum(
                colour.SDS_COLOURCHECKERS["BabelColor Average"]["white 9.5 (.05 D)"]
                * 0.025
            ),
        },
    )

    ET.SubElement(bsdf, "float", attrib={"name": "int_ior", "value": "1.46"})

    ET.SubElement(bsdf, "float", attrib={"name": "alpha", "value": "0.4"})

    with open(os.path.join(output_directory, "bsdfs_support.xml"), "w") as xml_file:
        xml_file.write(
            xml.dom.minidom.parseString(ET.tostring(scene)).toprettyxml(indent=" " * 4)
        )


_K_f = np.trapz(
    colour.SDS_ILLUMINANTS["E"].copy().align(MITSUBA_SHAPE).values,
    MITSUBA_SHAPE.range(),
)

_K_f_s = np.array([_K_f / 100, _K_f / 10, _K_f, _K_f * 10, _K_f * 100])


def export_emitters_files(K_f_s=_K_f_s, output_directory="include"):
    led_database = colour_datasets.load(
        "Measured Commercial LED Spectra - Brendel (2020)"
    )

    scene = ET.Element("scene", attrib={"version": "2.0.0"})

    for category, sds in [
        ("illuminant", colour.SDS_ILLUMINANTS),
        ("light_source", dict(colour.SDS_LIGHT_SOURCES, **led_database)),
    ]:
        for sd in sds.values():
            for K_f in K_f_s:
                name = f"{category}_{slugify(sd.name)}_{int(K_f)}K_f"

                emitter = ET.SubElement(
                    scene, "emitter", attrib={"type": "area", "id": name}
                )

                # K_f is used to normalize a light source to produce the same
                # amount of relative power per unit area per unit steradian that of
                # Illuminant E.
                if K_f != 1:
                    sd_K = sd.copy().align(MITSUBA_SHAPE)
                    K_n = np.trapz(sd_K.values, sd_K.wavelengths)
                else:
                    K_n = 1

                ET.SubElement(
                    emitter,
                    "spectrum",
                    attrib={
                        "name": "radiance",
                        "value": format_spectrum(
                            sd / K_n * K_f if category == "light_source" else sd
                        ),
                    },
                )

    with open(os.path.join(output_directory, "emitters.xml"), "w") as xml_file:
        xml_file.write(
            xml.dom.minidom.parseString(ET.tostring(scene)).toprettyxml(indent=" " * 4)
        )


def export_synthetic_LEDs(
    wavelengths=np.arange(400, 701, 1),
    fwhm=[10, 20, 30],
    K_f_s=_K_f_s / 10,
    normalise=True,
    output_directory="include",
):
    scene = ET.Element("scene", attrib={"version": "2.0.0"})
    for i in fwhm:
        luminous_flux = colour.luminous_flux(
            colour.sd_single_led(555, i).align(MITSUBA_SHAPE)
        )
        for j in wavelengths:
            sd = colour.sd_single_led(j, i).align(MITSUBA_SHAPE)
            if normalise:
                sd = scale_sd_to_luminous_flux(sd, luminous_flux)

            for K_f in K_f_s:
                name = f"light_source_{slugify(sd.name)}_{int(K_f)}K_f"
                emitter = ET.SubElement(
                    scene, "emitter", attrib={"type": "area", "id": name}
                )

                # K_f is used to normalize a light source to produce the same
                # amount of relative power per unit area per unit steradian that
                # of Illuminant E.
                if K_f != 1 and not normalise:
                    sd_K = sd.copy().align(MITSUBA_SHAPE)
                    K_n = np.trapz(sd_K.values, sd_K.wavelengths)
                else:
                    K_n = 1

                ET.SubElement(
                    emitter,
                    "spectrum",
                    attrib={
                        "name": "radiance",
                        "value": format_spectrum(sd / K_n * K_f),
                    },
                )

    with open(
        os.path.join(output_directory, "emitters_synthetic_leds.xml"), "w"
    ) as xml_file:
        xml_file.write(
            xml.dom.minidom.parseString(ET.tostring(scene)).toprettyxml(indent=" " * 4)
        )


def export_emitters_bt2020(
    wavelengths=[630, 532, 467],
    fwhm=1,
    K_f_s=_K_f_s,
    output_directory="include",
):
    sds = []

    sds += [colour.sd_single_led(i, fwhm).align(MITSUBA_SHAPE) for i in wavelengths]

    scene = ET.Element("scene", attrib={"version": "2.0.0"})
    for sd in sds:
        for K_f in K_f_s:
            name = f"light_source_bt2020_{slugify(sd.name)}_{int(K_f)}K_f"
            emitter = ET.SubElement(
                scene, "emitter", attrib={"type": "area", "id": name}
            )

            # K_f is used to normalize a light source to produce the same
            # amount of relative power per unit area per unit steradian that of
            # Illuminant E.
            if K_f != 1:
                sd_K = sd.copy().align(MITSUBA_SHAPE)
                K_n = np.trapz(sd_K.values, sd_K.wavelengths)
            else:
                K_n = 1
            
            ET.SubElement(
                emitter,
                "spectrum",
                attrib={"name": "radiance", "value": format_spectrum(sd / K_n * K_f)},
            )

    with open(
        os.path.join(output_directory, "emitters_synthetic_bt2020.xml"), "w"
    ) as xml_file:
        xml_file.write(
            xml.dom.minidom.parseString(ET.tostring(scene)).toprettyxml(indent=" " * 4)
        )


if __name__ == "__main__":
    export_AMPAS_training_data_bsdfs_files()
    export_AMPAS_training_data_bsdfs_files("plastic")
    export_AMPAS_training_data_bsdfs_files("roughplastic")

    export_colorchecker_classic_bsdfs_files()
    export_colorchecker_classic_bsdfs_files("ColorChecker N Ohta")
    export_colorchecker_classic_support_bsdfs_file()

    export_emitters_files()
    export_synthetic_LEDs()
    export_emitters_bt2020()
