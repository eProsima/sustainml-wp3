# SustainML – Work Package 3 (WP3)

[![SustainML](.figures/SustainML_GitHub.png)](https://sustainml.eu/)

<br>

<div class="menu" align="center">
    <strong>
        <a href="https://eprosima.com/index.php/downloads-all">Download</a>
        <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
        <a href="https://sustainml.readthedocs.io/en/latest">Docs</a>
        <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
        <a href="https://sustainml.eu/showroom/news">News</a>
        <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
        <a href="https://www.linkedin.com/company/eprosima/">LinkedIn</a>
        <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
        <a href="mailto:info@eprosima.com">Contact Us</a>
    </strong>
</div>

<br><br>

<div class="badges" align="center">
    <a href="https://github.com/eProsima/sustainml-wp3/releases"><img alt="Releases" src="https://img.shields.io/github/v/release/eProsima/sustainml-wp3?sort=semver"/></a>
    <a href="https://github.com/eProsima/sustainml-wp3/issues"><img alt="Issues" src="https://img.shields.io/github/issues/eProsima/sustainml-wp3.svg"/></a>
    <a href="https://github.com/eProsima/sustainml-wp3/network/members"><img alt="Forks" src="https://img.shields.io/github/forks/eProsima/sustainml-wp3.svg"/></a>
    <a href="https://github.com/eProsima/sustainml-wp3/stargazers"><img alt="Stars" src="https://img.shields.io/github/stars/eProsima/sustainml-wp3.svg"/></a>
    <a href="https://SustainML.readthedocs.io/en/latest/"><img alt="Documentation Status" src="https://img.shields.io/readthedocs/sustainml/latest"/></a>
</div>

<br><br>

This repository contains components developed as part of **Work Package 3 (WP3)** of the SustainML project.

WP3 focuses on assessing the **sustainability impact of machine learning models**: the energy consumption and the
carbon footprint of their inferences.

## Scope

This repository contains the **Carbon Footprint** node of the SustainML Framework (`carbon_footprint_node.py`):

- It computes the energy of one inference of the model from the power and latency given by the HW Resources Provider
  node of WP2.
- It turns that energy into a carbon footprint with the carbon intensity of the electricity grid of the country chosen
  by the user, or of the country detected from the public IP address of the machine (`carbon_intensities.csv`, the
  latest yearly values from Our World in Data, updated with `update_carbon_intensities.py`).
- When the HW Resources Provider gives no energy for a Hugging Face model, it measures the energy of running the model
  on the machine with CarbonTracker.
- It checks the result against the desired carbon footprint of the problem, for the automatic reiteration of the
  framework.

## Usage

The WP3 nodes are not intended to operate independently: they are integrated into the SustainML pipeline as part of
the `sustainml_modules` package of the [SustainML Library](https://github.com/eProsima/SustainML-Library), where this
repository is a Git submodule.

For installation, configuration, and execution instructions, refer to the main framework repository:

https://github.com/eProsima/SustainML-Framework
