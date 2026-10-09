# Copyright 2026 SustainML Consortium
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Regenerate carbon_intensities.csv: latest yearly grid carbon intensity of each country.

Source: Our World in Data, "Carbon intensity of electricity" (data from Ember),
https://ourworldindata.org/grapher/carbon-intensity-electricity

Countries: the UN member states plus Palestine (UN observer state), except those in
EXCLUDED, that have data. The carbon footprint node looks the intensity up by the
ISO alpha-2 code of the country chosen by the user, or detected from the IP address.
A last row, code WORLD, holds the world average: the fallback when the country is
unknown or not in the list (it is not a country, so the GUI list leaves it out).

Usage: python update_carbon_intensities.py
"""

import csv
import io
import os
import urllib.request

URL = ("https://ourworldindata.org/grapher/carbon-intensity-electricity.csv"
       "?v=1&csvType=full&useColumnShortNames=true")
OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "carbon_intensities.csv")

# ISO alpha-3 -> alpha-2 of the 193 UN member states and Palestine
COUNTRIES = {
    "AFG": "AF", "ALB": "AL", "DZA": "DZ", "AND": "AD", "AGO": "AO", "ATG": "AG", "ARG": "AR", "ARM": "AM",
    "AUS": "AU", "AUT": "AT", "AZE": "AZ", "BHS": "BS", "BHR": "BH", "BGD": "BD", "BRB": "BB", "BLR": "BY",
    "BEL": "BE", "BLZ": "BZ", "BEN": "BJ", "BTN": "BT", "BOL": "BO", "BIH": "BA", "BWA": "BW", "BRA": "BR",
    "BRN": "BN", "BGR": "BG", "BFA": "BF", "BDI": "BI", "CPV": "CV", "KHM": "KH", "CMR": "CM", "CAN": "CA",
    "CAF": "CF", "TCD": "TD", "CHL": "CL", "CHN": "CN", "COL": "CO", "COM": "KM", "COG": "CG", "CRI": "CR",
    "CIV": "CI", "HRV": "HR", "CUB": "CU", "CYP": "CY", "CZE": "CZ", "PRK": "KP", "COD": "CD", "DNK": "DK",
    "DJI": "DJ", "DMA": "DM", "DOM": "DO", "ECU": "EC", "EGY": "EG", "SLV": "SV", "GNQ": "GQ", "ERI": "ER",
    "EST": "EE", "SWZ": "SZ", "ETH": "ET", "FJI": "FJ", "FIN": "FI", "FRA": "FR", "GAB": "GA", "GMB": "GM",
    "GEO": "GE", "DEU": "DE", "GHA": "GH", "GRC": "GR", "GRD": "GD", "GTM": "GT", "GIN": "GN", "GNB": "GW",
    "GUY": "GY", "HTI": "HT", "HND": "HN", "HUN": "HU", "ISL": "IS", "IND": "IN", "IDN": "ID", "IRN": "IR",
    "IRQ": "IQ", "IRL": "IE", "ISR": "IL", "ITA": "IT", "JAM": "JM", "JPN": "JP", "JOR": "JO", "KAZ": "KZ",
    "KEN": "KE", "KIR": "KI", "KWT": "KW", "KGZ": "KG", "LAO": "LA", "LVA": "LV", "LBN": "LB", "LSO": "LS",
    "LBR": "LR", "LBY": "LY", "LIE": "LI", "LTU": "LT", "LUX": "LU", "MDG": "MG", "MWI": "MW", "MYS": "MY",
    "MDV": "MV", "MLI": "ML", "MLT": "MT", "MHL": "MH", "MRT": "MR", "MUS": "MU", "MEX": "MX", "FSM": "FM",
    "MDA": "MD", "MCO": "MC", "MNG": "MN", "MNE": "ME", "MAR": "MA", "MOZ": "MZ", "MMR": "MM", "NAM": "NA",
    "NRU": "NR", "NPL": "NP", "NLD": "NL", "NZL": "NZ", "NIC": "NI", "NER": "NE", "NGA": "NG", "MKD": "MK",
    "NOR": "NO", "OMN": "OM", "PAK": "PK", "PLW": "PW", "PAN": "PA", "PNG": "PG", "PRY": "PY", "PER": "PE",
    "PHL": "PH", "POL": "PL", "PRT": "PT", "QAT": "QA", "KOR": "KR", "ROU": "RO", "RUS": "RU", "RWA": "RW",
    "KNA": "KN", "LCA": "LC", "VCT": "VC", "WSM": "WS", "SMR": "SM", "STP": "ST", "SAU": "SA", "SEN": "SN",
    "SRB": "RS", "SYC": "SC", "SLE": "SL", "SGP": "SG", "SVK": "SK", "SVN": "SI", "SLB": "SB", "SOM": "SO",
    "ZAF": "ZA", "SSD": "SS", "ESP": "ES", "LKA": "LK", "SDN": "SD", "SUR": "SR", "SWE": "SE", "CHE": "CH",
    "SYR": "SY", "TJK": "TJ", "THA": "TH", "TLS": "TL", "TGO": "TG", "TON": "TO", "TTO": "TT", "TUN": "TN",
    "TUR": "TR", "TKM": "TM", "TUV": "TV", "UGA": "UG", "UKR": "UA", "ARE": "AE", "GBR": "GB", "TZA": "TZ",
    "USA": "US", "URY": "UY", "UZB": "UZ", "VUT": "VU", "VEN": "VE", "VNM": "VN", "YEM": "YE", "ZMB": "ZM",
    "ZWE": "ZW",
    "PSE": "PS",
}

# Countries left out of the list
EXCLUDED = {"ISR"}

WORLD = "OWID_WRL"


def main():
    request = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        rows = csv.DictReader(io.StringIO(response.read().decode("utf-8")))
        latest = {}
        for row in rows:
            code = row["code"]
            if (code not in COUNTRIES and code != WORLD) or code in EXCLUDED or row["co2_intensity__gco2_kwh"] == "":
                continue
            if code not in latest or int(row["year"]) > int(latest[code]["year"]):
                latest[code] = row

    with open(OUTPUT, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["alpha-2", "Entity", "Code", "Year", "Carbon intensity of electricity (gCO2/kWh)"])
        for code in sorted((c for c in latest if c != WORLD), key=lambda c: COUNTRIES[c]):
            row = latest[code]
            writer.writerow([COUNTRIES[code], row["entity"], code, row["year"], row["co2_intensity__gco2_kwh"]])
        world = latest[WORLD]
        writer.writerow(["WORLD", "World average", WORLD, world["year"], world["co2_intensity__gco2_kwh"]])
    print(f"Wrote {len(latest) - 1} countries and the world average to {OUTPUT}")
    without_data = sorted(c for c in COUNTRIES if c not in latest and c not in EXCLUDED)
    print(f"No data for: {', '.join(without_data)}")


if __name__ == "__main__":
    main()
