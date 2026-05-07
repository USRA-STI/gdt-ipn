#  CONTAINS TECHNICAL DATA/COMPUTER SOFTWARE DELIVERED TO THE U.S. GOVERNMENT WITH UNLIMITED RIGHTS
#
#  Contract No.: CA 80NSSC24M0035
#  Contractor Name: Universities Space Research Association
#  Contractor Address: 7178 Columbia Gateway Drive, Columbia, MD 21046
#
#  Copyright 2021-2025 by Universities Space Research Association (USRA). All rights reserved.
#
#  Original IPN development funded through FY21 USRA Internal Research and Development Funds
#  and FY21 NASA-MSFC Center Innovation Funds
#
#  IPN code developed by:
#
#                Corinne Fletcher, Rachel Hamburg and Adam Goldstein
#                Universities Space Research Association
#                Science and Technology Institute
#                https://sti.usra.edu
#
#                Peter Veres
#                University of Alabama in Huntsville
#                Huntsville, AL
#
#                Michelle Hui
#                National Aeronautics and Space Administration (NASA)
#                Marshall Space Flight Center
#                Astrophysics Branch (ST-12)
#
#
#  With code contributions by:
#
#                Dmitry Svinkin
#                Ioffe Institute
#                St. Petersburg, Russia
#
#  Included in the Gamma-Ray Data Toolkit
#  Copyright 2017-2025 by Universities Space Research Association (USRA). All rights reserved.
#
#  Developed by: William Cleveland and Adam Goldstein
#                Universities Space Research Association
#                Science and Technology Institute
#                https://sti.usra.edu
#
#  Developed by: Daniel Kocevski
#                National Aeronautics and Space Administration (NASA)
#                Marshall Space Flight Center
#                Astrophysics Branch (ST-12)
#
#  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
#  in compliance with the License. You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software distributed under the License
#  is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
#  implied. See the License for the specific language governing permissions and limitations under the
#  License.

from unittest import TestCase
import numpy as np
import astropy.coordinates as a_coords
import astropy.units as u
from astropy.time import Time
from gdt.ipn.algorithms.annulus import Annulus
from gdt.ipn.algorithms.ccf import Ipn
from gdt.ipn.instrument import *

class TestSpacecraftPosition(TestCase):

        def setUp(self):
            self.pos = SpacecraftPosition()


        def test_unit(self):
           self.assertAlmostEqual(self.pos.unit,'km')

        def test_vector(self):
            x,y,z = self.pos.vector
            self.assertAlmostEqual(x, 0.0)
            self.assertAlmostEqual(y, 0.0)
            self.assertAlmostEqual(z, 0.0)

        def test_vector_err(self):
            x_err,y_err,z_err = self.pos.vector_err
            self.assertAlmostEqual(x_err, 0.0)
            self.assertAlmostEqual(y_err, 0.0)
            self.assertAlmostEqual(z_err, 0.0)

        def test_origin_distance(self):
            dist = self.pos.origin_distance
            self.assertAlmostEqual(dist, 0.0)

        def test_from_distance(self):
            spacepos = self.pos.from_distance(5000, 50, unit='km')

            x,y,z = spacepos.vector
            self.assertAlmostEqual(x, 2886.751345, places=5)
            self.assertAlmostEqual(y, 2886.751345, places=5)
            self.assertAlmostEqual(z, 2886.751345, places=5)

            x_err,y_err,z_err = spacepos.vector_err
            self.assertAlmostEqual(x_err, 50.0, places=2)
            self.assertAlmostEqual(y_err, 50.0, places=2)
            self.assertAlmostEqual(z_err, 50.0, places=2)

            dist = spacepos.origin_distance
            self.assertAlmostEqual(dist, 5000.0, places=2)

            err = spacepos.origin_distance_uncertainty
            self.assertAlmostEqual(err, 50.0, places=2)

            unit_x, unit_y, unit_z = spacepos.unit_vector
            self.assertAlmostEqual(unit_x, 0.57735, places=5)
            self.assertAlmostEqual(unit_y, 0.57735, places=5)
            self.assertAlmostEqual(unit_z, 0.57735, places=5)

            unit_x_err, unit_y_err, unit_z_err = spacepos.unit_vector_uncertainty
            self.assertAlmostEqual(unit_x_err, 0.0115470, places=5)
            self.assertAlmostEqual(unit_y_err, 0.0115470, places=5)
            self.assertAlmostEqual(unit_z_err, 0.0115470, places=5)

        def test_from_vectors(self):
            vecpos = self.pos.from_vectors((5500,6500,7500),(50,60,70), unit='m')

            self.assertAlmostEqual(vecpos.unit,'m')

            x,y,z = vecpos.vector
            self.assertAlmostEqual(x, 5500, places=5)
            self.assertAlmostEqual(y, 6500, places=5)
            self.assertAlmostEqual(z, 7500, places=5)

            x_err,y_err,z_err = vecpos.vector_err
            self.assertAlmostEqual(x_err, 50.0, places=2)
            self.assertAlmostEqual(y_err, 60.0, places=2)
            self.assertAlmostEqual(z_err, 70.0, places=2)

            dist = vecpos.origin_distance
            self.assertAlmostEqual(dist, 11346.805717, places=5)

            err = vecpos.origin_distance_uncertainty
            self.assertAlmostEqual(err, 62.5261110, places=6)

            unit_x, unit_y, unit_z = vecpos.unit_vector
            self.assertAlmostEqual(unit_x, 0.4847179, places=6)
            self.assertAlmostEqual(unit_y, 0.572848, places=6)
            self.assertAlmostEqual(unit_z, 0.6609789, places=6)

            unit_x_err, unit_y_err, unit_z_err = vecpos.unit_vector_uncertainty
            self.assertAlmostEqual(unit_x_err, 0.005152, places=5)
            self.assertAlmostEqual(unit_y_err, 0.006158, places=5)
            self.assertAlmostEqual(unit_z_err, 0.007164, places=5)

        def test_to_units(self):
            spacepos = self.pos.from_distance(5000, 50, unit='km')

            new_units=spacepos.to_units(unit='m')
            self.assertAlmostEqual(new_units.unit,'m')

            x,y,z = new_units.vector
            self.assertAlmostEqual(x, 2886751.345, places=2)
            self.assertAlmostEqual(y, 2886751.345, places=2)
            self.assertAlmostEqual(z, 2886751.345, places=2)

            x_err,y_err,z_err = new_units.vector_err
            self.assertAlmostEqual(x_err, 50000.0, places=2)
            self.assertAlmostEqual(y_err, 50000.0, places=2)
            self.assertAlmostEqual(z_err, 50000.0, places=2)

            dist = new_units.origin_distance
            self.assertAlmostEqual(dist, 5000000.0, places=2)

            err = new_units.origin_distance_uncertainty
            self.assertAlmostEqual(err, 50000.0, places=2)

            unit_x, unit_y, unit_z = new_units.unit_vector
            self.assertAlmostEqual(unit_x, 0.57735, places=5)
            self.assertAlmostEqual(unit_y, 0.57735, places=5)
            self.assertAlmostEqual(unit_z, 0.57735, places=5)

            unit_x_err, unit_y_err, unit_z_err = new_units.unit_vector_uncertainty
            self.assertAlmostEqual(unit_x_err, 0.0115470, places=5)
            self.assertAlmostEqual(unit_y_err, 0.0115470, places=5)
            self.assertAlmostEqual(unit_z_err, 0.0115470, places=5)


        def test_distance(self):
            instrument1 = self.pos.from_distance(5000, 50, unit='km')
            instrument2 = self.pos.from_vectors((5500,6500,7500),(50,60,70), unit='m')

            dist1 =instrument1.distance(instrument2)
            self.assertAlmostEqual(dist1, 4988.741870, places=5)

            dist2 =instrument2.distance(instrument1)
            self.assertAlmostEqual(dist2, 4988741.87020, places=5)

            new_units = instrument2.to_units('km')
            dist_new=new_units.distance(instrument1)
            self.assertAlmostEqual(dist_new, dist1, places=1)

        def test_distance_uncertainty(self):
            instrument1 = self.pos.from_distance(5000, 50, unit='km')
            instrument2 = self.pos.from_vectors((5500,6500,7500),(50,60,70), unit='m')

            dist_unc1 =instrument1.distance_uncertainty(instrument2)
            self.assertAlmostEqual(dist_unc1, 50.0000366, places=5)

            dist_unc2 =instrument2.distance_uncertainty(instrument1)
            self.assertAlmostEqual(dist_unc2, 50000.03666, places=5)

        def test_baseline_uncertainty(self):
            instrument1 = self.pos.from_distance(5000, 50, unit='km')
            instrument2 = self.pos.from_vectors((5500,6500,7500),(50,60,70), unit='m')

            ra_unc1, dec_unc1=instrument1.baseline_uncertainty(instrument2)
            self.assertAlmostEqual(ra_unc1, 0.010023, places=5)
            self.assertAlmostEqual(dec_unc1, 0.012273, places=5)


            ra_unc2, dec_unc2 =instrument2.baseline_uncertainty(instrument1)
            self.assertAlmostEqual(ra_unc2, 0.010023, places=5)
            self.assertAlmostEqual(dec_unc2, 0.012273, places=5)

        def test_baseline(self):
            instrument1 = self.pos.from_distance(5000, 50, unit='km')
            instrument2 = self.pos.from_vectors((5500,6500,7500),(50,60,70), unit='m')

            ra1, dec1 = instrument1.baseline(instrument2)
            self.assertAlmostEqual(ra1, 0.785225, places=5)
            self.assertAlmostEqual(dec1, 0.615234, places=5)

            ra2, dec2 =instrument2.baseline(instrument1)
            self.assertAlmostEqual(ra2, 3.926817, places=5)
            self.assertAlmostEqual(dec2, -0.615234, places=5)

        def test_geocenter_correction(self):
            geo_corr = self.pos.geocenter_correction(195,230, time_unit='s')
            self.assertAlmostEqual(geo_corr, 0.0, places=5)

            instrument1 = self.pos.from_vectors((5500,6500,7500),(50,60,70), unit='m')
            geo_corr1 = instrument1.geocenter_correction(100,230, time_unit='s')
            self.assertAlmostEqual(geo_corr1, 3.0841e-05, places=5)

            instrument2 = self.pos.from_distance(5000, 50, unit='km')
            geo_corr2 = instrument2.geocenter_correction(60,100, time_unit = 's')
            self.assertAlmostEqual(geo_corr2, -0.0071987, places=5)


class TestSpacecraft(TestCase):

        def setUp(self):
            self.sc1 = Spacecraft(
                SpacecraftPosition.from_vectors((1000.0, 2000.0, 3000.0),
                                                (1.0, 1.0, 1.0),
                                                unit='km'))
            self.sc2 = Spacecraft(
                SpacecraftPosition.from_vectors((4000.0, -500.0, 600.0),
                                                (1.0, 1.0, 1.0),
                                                unit='km'))

        def test_to_spacecraft_frame(self):
            obstime = Time('2024-01-01T00:00:00', scale='utc')
            frame = self.sc1.to_spacecraft_frame(obstime)

            self.assertIsInstance(frame, SpacecraftFrame)
            self.assertAlmostEqual(frame.obsgeoloc.x.to_value(u.km), 1000.0)
            self.assertAlmostEqual(frame.obsgeoloc.y.to_value(u.km), 2000.0)
            self.assertAlmostEqual(frame.obsgeoloc.z.to_value(u.km), 3000.0)

        def test_barycentric_position(self):
            obstime = Time('2024-01-01T00:00:00', scale='utc')
            bary_pos = self.sc1.barycentric_position(obstime)

            earth_bary = a_coords.get_body_barycentric('earth', obstime.tdb)
            expected = earth_bary.xyz.to_value(u.km) + self.sc1.position.vector

            self.assertIsInstance(bary_pos, SpacecraftPosition)
            self.assertEqual(bary_pos.unit, 'km')
            np.testing.assert_allclose(bary_pos.vector, expected, atol=1e-3)

        def test_baseline_to_barycentric_uses_separate_times(self):
            # sc_a at +x, sc_b at -x: geocentric baseline points along +x (RA=0, Dec=0)
            sc_a = Spacecraft(SpacecraftPosition.from_vectors((7000.0, 0.0, 0.0),
                                                             (0.0, 0.0, 0.0),
                                                             unit='km'))
            sc_b = Spacecraft(SpacecraftPosition.from_vectors((-7000.0, 0.0, 0.0),
                                                             (0.0, 0.0, 0.0),
                                                             unit='km'))

            ra_geo, dec_geo = sc_a.baseline_to(sc_b)
            ra_bary, dec_bary = sc_a.baseline_to(
                sc_b,
                obstime=Time('2024-01-01T00:00:00', scale='utc'),
                other_obstime=Time('2024-01-02T00:00:00', scale='utc'),
                barycentric=True)

            self.assertAlmostEqual(ra_geo, 0.0, places=7)
            self.assertAlmostEqual(dec_geo, 0.0, places=7)
            self.assertFalse(np.isnan(ra_bary))
            self.assertFalse(np.isnan(dec_bary))
            self.assertGreater(abs(ra_bary) + abs(dec_bary), 1e-6)

        def test_barycentric_common_reference_preserves_baseline(self):
            obstime = Time('2024-01-01T00:00:00', scale='utc')
            earth_ref = a_coords.get_body_barycentric('earth', obstime.tdb)
            earth_ref_vec = earth_ref.xyz.to_value(u.km)

            sc1_bary = self.sc1.barycentric_position(obstime)
            sc2_bary = self.sc2.barycentric_position(obstime)

            sc1_shifted = SpacecraftPosition.from_vectors(
                sc1_bary.vector - earth_ref_vec,
                self.sc1.position.vector_err,
                unit='km')
            sc2_shifted = SpacecraftPosition.from_vectors(
                sc2_bary.vector - earth_ref_vec,
                self.sc2.position.vector_err,
                unit='km')

            np.testing.assert_allclose(sc1_shifted.vector,
                                       self.sc1.position.vector,
                                       atol=1e-6)
            np.testing.assert_allclose(sc2_shifted.vector,
                                       self.sc2.position.vector,
                                       atol=1e-6)

            ra_geo, dec_geo = self.sc1.baseline_to(self.sc2)
            ra_shifted, dec_shifted = sc1_shifted.baseline(sc2_shifted)

            self.assertAlmostEqual(ra_shifted, ra_geo, places=7)
            self.assertAlmostEqual(dec_shifted, dec_geo, places=7)


class TestAnnulus(TestCase):

        def test_radius_is_independent_of_baseline_ra_quadrant(self):
            time_offset = TimeUncertainty(0.001, (0.0, 0.0), unit='s')

            sc_ref = Spacecraft(
                SpacecraftPosition.from_vectors((0.0, 0.0, 0.0),
                                                (0.0, 0.0, 0.0),
                                                unit='km'))
            sc_other_neg_y = Spacecraft(
                SpacecraftPosition.from_vectors((-1000.0, 100.0, 0.0),
                                                (0.0, 0.0, 0.0),
                                                unit='km'))
            sc_other_pos_y = Spacecraft(
                SpacecraftPosition.from_vectors((-1000.0, -100.0, 0.0),
                                                (0.0, 0.0, 0.0),
                                                unit='km'))

            annulus_neg_y = Annulus(sc_ref, sc_other_neg_y, time_offset)
            annulus_pos_y = Annulus(sc_ref, sc_other_pos_y, time_offset)

            distance = sc_ref.position.distance(sc_other_neg_y.position)
            expected = np.rad2deg(
                np.arccos(np.clip(-299792.458 * time_offset.dt / distance,
                                  -1.0, 1.0)))

            self.assertAlmostEqual(annulus_neg_y.radius(), expected, places=7)
            self.assertAlmostEqual(annulus_pos_y.radius(), expected, places=7)


class TestIpn(TestCase):

        def test_get_healpix_barycentric_matches_manual_annulus(self):
            sc1 = Spacecraft(
                SpacecraftPosition.from_vectors((7000.0, 0.0, 0.0),
                                                (1.0, 1.0, 1.0),
                                                unit='km'),
                time_uncert=0.002)
            sc2 = Spacecraft(
                SpacecraftPosition.from_vectors((-2.0e6, 2.5e5, 1.0e5),
                                                (1.0, 1.0, 1.0),
                                                unit='km'),
                time_uncert=0.003)

            ipn = Ipn.from_list([sc1, sc2])
            ipn._time_offset = TimeUncertainty(4.5, (0.2, 0.2), unit='s')

            obstime = Time('2024-01-01T00:00:00', scale='utc')
            other_obstime = Time('2024-01-01T00:00:05', scale='utc')

            hpx = ipn.get_healpix(nside=32,
                                  obstime=obstime,
                                  other_obstime=other_obstime,
                                  barycentric=True)

            sc1_bary = Spacecraft(sc1.barycentric_position(obstime),
                                  time_uncert=sc1.time_uncert.err,
                                  time_units=sc1.time_uncert.unit)
            sc2_bary = Spacecraft(sc2.barycentric_position(other_obstime),
                                  time_uncert=sc2.time_uncert.err,
                                  time_units=sc2.time_uncert.unit)
            annulus = Annulus(sc1_bary, sc2_bary, ipn.time_offset)
            expected = Ipn.get_healpix.__globals__['IpnHealPixLocalization'].from_annulus(
                *annulus.center(), annulus.radius(), annulus.total_width(), nside=32)

            np.testing.assert_allclose(hpx.prob, expected.prob)
