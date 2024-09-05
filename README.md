# Hand-drawn Circuit Recognizer

This repository provides a solution for recognizing and interpreting hand-drawn circuits using Python and OpenCV. The project involves building a system that can automatically process and analyze circuit diagrams drawn by hand, converting them into digital schematics.

## Overview

The Hand-drawn Circuit Recognizer utilizes advanced computer vision techniques to identify and classify various components of a circuit from an image. The system is designed to handle irregularities and variations typically found in hand-drawn diagrams, enabling efficient and accurate circuit recognition.

## Features

- **Segmentation and Classification**: Implements state-of-the-art segmentation techniques to isolate circuit symbols and uses classification algorithms to identify components such as resistors, capacitors, and connectors.
- **Symbol Extraction and Node Construction**: Extracts individual symbols from circuit wires and constructs nodes to enhance the accuracy of component classification and interconnection analysis.
- **Netlist Generation**: Converts recognized symbols and their connections into a netlist format, which represents the circuit in a structured manner suitable for further analysis or digital schematic creation.

## Tools and Technologies

- **Python**: Programming language used for developing the recognition algorithms and overall system logic.
- **OpenCV**: Library used for image processing, including segmentation, feature extraction, and classification tasks.

## Installation

To set up the project locally, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/project-b.git
    ```
2. Navigate to the project directory:
    ```bash
    cd project-b
    ```
3. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Place your hand-drawn circuit images in the `input_images` directory.
2. Run the recognition script:
    ```bash
    python recognize_circuit.py
    ```
3. The output, including classified symbols and netlists, will be saved in the `output` directory.


