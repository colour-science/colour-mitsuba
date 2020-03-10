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

import colour
import colour_datasets
import xml.etree.ElementTree as ET

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2019-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'MITSUBA_SHAPE', 'slugify', 'format_spectrum',
    'export_AMPAS_training_data_bsdfs_files',
    'export_colorchecker_classic_bsdfs_files',
    'export_colorchecker_classic_support_bsdfs_file', 'export_emitters_files',
    'export_synthetic_LEDs'
]

MITSUBA_SHAPE = colour.SpectralShape(360, 830, 5)


def slugify(a):
    return re.sub(r'\s|-|\.', '_',
                  re.sub(r'(?u)[^-\w.]', ' ',
                         str(a).strip()).strip().lower())


def format_spectrum(sd, decimals=15):
    return ', '.join('{1}:{2:0.{0}f}'.format(decimals, wavelength, value)
                     for wavelength, value in zip(sd.wavelengths, sd.values))


def export_AMPAS_training_data_bsdfs_files(bsdf_type='diffuse',
                                           ior=1.460,
                                           alpha=0.05,
                                           output_directory='include'):
    sensitivities_database = colour_datasets.load('RAW to ACES Utility Data')

    training_data = sensitivities_database['training']['190-patch']

    scene = ET.Element('scene', attrib={'version': '2.0.0'})
    for sd in colour.colorimetry.sds_and_multi_sds_to_sds(training_data):
        bsdf = ET.SubElement(
            scene,
            'bsdf',
            attrib={
                'type': bsdf_type,
                'id': sd.name.split('-')[0].strip()
            })

        ET.SubElement(
            bsdf,
            'spectrum',
            attrib={
                'name': ('reflectance'
                         if bsdf_type == 'diffuse' else 'diffuse_reflectance'),
                'value':
                format_spectrum(sd)
            })

        if bsdf_type in ('plastic', 'roughplastic'):
            ET.SubElement(
                bsdf, 'float', attrib={
                    'name': 'int_ior',
                    'value': str(ior)
                })

        if bsdf_type == 'roughplastic':
            ET.SubElement(
                bsdf, 'float', attrib={
                    'name': 'alpha',
                    'value': str(alpha)
                })

    with open(
            os.path.join(output_directory,
                         'bsdfs_190_patch_{}.xml'.format(bsdf_type)),
            'w') as xml_file:

        xml_file.write(
            xml.dom.minidom.parseString(
                ET.tostring(scene)).toprettyxml(indent=' ' * 4))


def export_colorchecker_classic_bsdfs_files(
        colour_checker='BabelColor Average',
        output_directory='colorchecker_classic/include'):
    scene = ET.Element('scene', attrib={'version': '2.0.0'})
    for sd in colour.COLOURCHECKERS_SDS[colour_checker].values():
        bsdf = ET.SubElement(
            scene,
            'bsdf',
            attrib={
                'type': 'diffuse',
                'id': 'colorchecker_classic_{0}'.format(slugify(sd.name))
            })

        ET.SubElement(
            bsdf,
            'spectrum',
            attrib={
                'name': 'reflectance',
                'value': format_spectrum(sd)
            })

    with open(
            os.path.join(output_directory, 'bsdfs_{0}.xml'.format(
                slugify(colour_checker))), 'w') as xml_file:

        xml_file.write(
            xml.dom.minidom.parseString(
                ET.tostring(scene)).toprettyxml(indent=' ' * 4))


def export_colorchecker_classic_support_bsdfs_file(
        output_directory='colorchecker_classic/include'):
    scene = ET.Element('scene', attrib={'version': '2.0.0'})
    bsdf = ET.SubElement(
        scene,
        'bsdf',
        attrib={
            'type': 'roughplastic',
            'id': 'colorchecker_classic_support'
        })

    ET.SubElement(
        bsdf,
        'spectrum',
        attrib={
            'name':
            'diffuse_reflectance',
            'value':
            format_spectrum(colour.COLOURCHECKERS_SDS['BabelColor Average']
                            ['white 9.5 (.05 D)'] * 0.025)
        })

    ET.SubElement(bsdf, 'float', attrib={'name': 'int_ior', 'value': '1.46'})

    ET.SubElement(bsdf, 'float', attrib={'name': 'alpha', 'value': '0.4'})

    with open(os.path.join(output_directory, 'bsdfs_support.xml'),
              'w') as xml_file:

        xml_file.write(
            xml.dom.minidom.parseString(
                ET.tostring(scene)).toprettyxml(indent=' ' * 4))


_K_f = np.trapz(colour.ILLUMINANTS_SDS['E'].copy().align(MITSUBA_SHAPE).values,
                MITSUBA_SHAPE.range())


def export_emitters_files(output_directory='include', K_f=_K_f):
    for category, sds in [('illuminant', colour.ILLUMINANTS_SDS),
                          ('light_source', colour.LIGHT_SOURCES_SDS)]:
        for sd in sds.values():
            name = '{0}_{1}'.format(category, slugify(sd.name))

            scene = ET.Element('scene', attrib={'version': '2.0.0'})

            emitter = ET.SubElement(
                scene, 'emitter', attrib={
                    'type': 'area',
                    'id': name
                })

            # K_f is used to normalize a light source to produce the same
            # amount of relative power per unit area per unit steradian that of
            # Illuminant E.
            if K_f != 1:
                sd_K = sd.copy().align(MITSUBA_SHAPE)
                K_n = np.trapz(sd_K.values, sd_K.wavelengths)

            ET.SubElement(
                emitter,
                'spectrum',
                attrib={
                    'name':
                    'radiance',
                    'value':
                    format_spectrum(sd / K_n *
                                    K_f if category == 'light_source' else sd)
                })

            with open(
                    os.path.join(output_directory,
                                 'emitter_{0}.xml'.format(name)),
                    'w') as xml_file:

                xml_file.write(
                    xml.dom.minidom.parseString(
                        ET.tostring(scene)).toprettyxml(indent=' ' * 4))


def export_synthetic_LEDs(
        wavelengths=np.arange(450, 650, 10),
        fwhm=20,
        output_directory='include',
        K_f=_K_f / 20):
    sds = [
        colour.sd_single_led(i, fwhm).align(MITSUBA_SHAPE) for i in wavelengths
    ]

    for sd in sds:
        name = 'light_source_{0}'.format(slugify(sd.name))
        scene = ET.Element('scene', attrib={'version': '2.0.0'})
        emitter = ET.SubElement(
            scene, 'emitter', attrib={
                'type': 'area',
                'id': name
            })

        # K_f is used to normalize a light source to produce the same
        # amount of relative power per unit area per unit steradian that of
        # Illuminant E.
        if K_f != 1:
            sd_K = sd.copy().align(MITSUBA_SHAPE)
            K_n = np.trapz(sd_K.values, sd_K.wavelengths)

        ET.SubElement(
            emitter,
            'spectrum',
            attrib={
                'name': 'radiance',
                'value': format_spectrum(sd / K_n * K_f)
            })

        with open(
                os.path.join(output_directory, 'emitter_{0}.xml'.format(name)),
                'w') as xml_file:

            xml_file.write(
                xml.dom.minidom.parseString(
                    ET.tostring(scene)).toprettyxml(indent=' ' * 4))


if __name__ == '__main__':
    export_AMPAS_training_data_bsdfs_files()
    export_AMPAS_training_data_bsdfs_files('plastic')
    export_AMPAS_training_data_bsdfs_files('roughplastic')

    export_colorchecker_classic_bsdfs_files()
    export_colorchecker_classic_bsdfs_files('ColorChecker N Ohta')
    export_colorchecker_classic_support_bsdfs_file()

    export_emitters_files()
    export_synthetic_LEDs()
