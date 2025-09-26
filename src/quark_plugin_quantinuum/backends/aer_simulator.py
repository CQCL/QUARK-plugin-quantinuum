from dataclasses import dataclass, field
from typing import override
import logging

from qiskit_aer import AerSimulator as QiskitAS
from qiskit import QuantumCircuit
from quark.core import Core, Data, Result
from quark.interface_types import Other

from .backend_input import BackendInput
from .backend_result import BackendResult

logger = logging.getLogger()


@dataclass
class AerSimulator(Core):

    n_shots: int = 100
    _results: BackendResult | None = field(init=False, default=None)

    @override
    def preprocess(self, input_data: Other[BackendInput]) -> Result:
        backend = QiskitAS()
        backend_input = input_data.data
        circuits = backend_input.circuits
        self.warn_on_large_circuits(circuits)

        counts_per_circuit = []
        logger.info(f"Running circuits on AerSimulator")
        for n, circuit in enumerate(circuits):
            logger.info(f"Running circuit for {n} Trotter steps")
            counts_per_circuit.append(backend.run(circuit, shots=self.n_shots).result().get_counts(circuit))

        self._results = BackendResult(
            counts=counts_per_circuit,
            n_shots=self.n_shots
        )
        return Data(None)


    @override
    def postprocess(self, input_data: Data) -> Result:
        return Data(Other(self._results))

    @staticmethod
    def warn_on_large_circuits(circuits: list[QuantumCircuit]) -> None:
        warning_n_qubits = 30
        max_n_qubit = max([circuit.num_qubits for circuit in circuits])
        if max_n_qubit > warning_n_qubits:
            logger.warning(f"Simulating circuits with over {warning_n_qubits} qubits. The high memory"
                           f" requirements can lead to memory errors on some systems.")
