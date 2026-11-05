"""
aggregation.py
================
Custom Flower (flwr) aggregation strategy: FedCosinePairwise.

Strategi FedAvg custom yang membobotkan kontribusi tiap client
berdasarkan PAIRWISE cosine similarity antar parameter model
seluruh client pada ronde tersebut (dihitung sebelum agregasi).

Cara pakai singkat:
    strategy = build_fedcosine_pairwise_strategy(
        num_clients=10,
        init_params=initial_parameters,
        evaluate_fn=my_evaluate_fn,
        fit_metrics_agg=my_fit_metrics_agg,
        temperature=0.1,
        min_weight=0.0,
        similarity_agg="mean",  # atau "median"
    )
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


# ============================================================
# Helper functions
# ============================================================

def _flatten(ndarrays: List[np.ndarray]) -> np.ndarray:
    """Ratakan (flatten) seluruh layer parameter model menjadi 1 vector."""
    return np.concatenate([arr.ravel() for arr in ndarrays])


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Hitung cosine similarity antara dua vector 1D."""
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


# ============================================================
# Builder function
# ============================================================

def build_fedcosine_pairwise_strategy(
    num_clients: int,
    init_params: Parameters,
    evaluate_fn,
    fit_metrics_agg,
    temperature: float = 0.1,
    min_weight: float = 0.0,
    similarity_agg: str = "mean",
    data_influence: float = 0.0,
) -> "FedCosinePairwise":
    """
    Args tambahan dibanding versi lama:
        similarity_agg:
            "mean"   -> skor client = rata-rata similarity ke semua
                        client lain (default).
            "median" -> skor client = median similarity ke semua
                        client lain. Lebih tahan terhadap client
                        outlier / mencurigakan karena tidak gampang
                        tertarik oleh nilai ekstrem.
        data_influence:
            Seberapa besar jumlah data (num_examples) tiap client
            ikut mempengaruhi bobot akhir, di antara similarity
            (prioritas utama) dan proporsi data.

            0.0  -> murni pairwise similarity (perilaku lama, default).
            1.0  -> murni proporsional ke jumlah data (seperti FedAvg biasa).
            0 < data_influence < 1 -> kombinasi, similarity TETAP jadi
                    faktor dominan karena digabung lewat weighted
                    geometric mean (perkalian di ruang log), bukan
                    dijumlah biasa. Nilai kecil (mis. 0.1-0.3) disarankan
                    kalau similarity ingin tetap jadi prioritas.
    """
    if not 0.0 <= data_influence <= 1.0:
        raise ValueError("data_influence harus di antara 0.0 dan 1.0")

    return FedCosinePairwise(
        temperature=temperature,
        min_weight=min_weight,
        similarity_agg=similarity_agg,
        data_influence=data_influence,
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=num_clients,
        min_available_clients=num_clients,
        initial_parameters=init_params,
        evaluate_fn=evaluate_fn,
        fit_metrics_aggregation_fn=fit_metrics_agg,
    )


# ============================================================
# FedCosine Pairwise Strategy
# ============================================================
class FedCosinePairwise(FedAvg):
    """
    Custom FedAvg dengan pembobotan berdasarkan PAIRWISE cosine
    similarity antar client (dihitung sebelum agregasi apapun).

    Mekanisme:

        1. Ambil parameter (atau feature extraction) tiap client.
        2. Hitung cosine similarity ANTAR SEMUA PASANGAN client
           -> matriks similarity N x N.
        3. Skor tiap client = mean/median similarity-nya terhadap
           semua client lain (bukan terhadap reference tunggal).
        4. Ubah skor jadi bobot pakai softmax.
        5. Lakukan weighted average per layer.
        6. Hasil agregasi jadi model global ronde ini.

    Args:
        temperature:
            Mengontrol ketajaman softmax, sama seperti versi lama.
        min_weight:
            Bobot minimum per client, sama seperti versi lama.
        similarity_agg:
            "mean" atau "median", lihat build_fedcosine_pairwise_strategy.
        data_influence:
            0.0-1.0, seberapa besar jumlah data client ikut menentukan
            bobot akhir tanpa menggeser prioritas dari similarity.
            Lihat build_fedcosine_pairwise_strategy untuk detail.
    """

    def __init__(
        self,
        *args,
        temperature: float = 0.1,
        min_weight: float = 0.0,
        similarity_agg: str = "mean",
        data_influence: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.temperature = temperature
        self.min_weight = min_weight

        if similarity_agg not in ("mean", "median"):
            raise ValueError("similarity_agg harus 'mean' atau 'median'")
        self.similarity_agg = similarity_agg

        if not 0.0 <= data_influence <= 1.0:
            raise ValueError("data_influence harus di antara 0.0 dan 1.0")
        self.data_influence = data_influence

        # Disimpan untuk keperluan logging/analisis dari luar kelas
        self.last_data_shares: Optional[np.ndarray] = None

        # Disimpan untuk keperluan logging/analisis dari luar kelas
        self.last_similarity_matrix: Optional[np.ndarray] = None

    # --------------------------------------------------------
    # Helper: matriks pairwise similarity
    # --------------------------------------------------------

    def _pairwise_similarity_matrix(
        self, thetas: List[np.ndarray]
    ) -> np.ndarray:
        """Hitung matriks cosine similarity N x N antar semua client."""
        n = len(thetas)
        sim_matrix = np.zeros((n, n), dtype=np.float64)

        for i in range(n):
            for j in range(i, n):
                if i == j:
                    sim_matrix[i, j] = 1.0
                else:
                    s = cosine_similarity(thetas[i], thetas[j])
                    sim_matrix[i, j] = s
                    sim_matrix[j, i] = s

        return sim_matrix

    # --------------------------------------------------------
    # Helper: skor representativitas tiap client
    # --------------------------------------------------------

    def _representativeness_scores(
        self, sim_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Skor tiap client = agregasi (mean/median) similarity-nya
        terhadap SEMUA client lain (similarity ke diri sendiri
        dibuang dulu).
        """
        n = sim_matrix.shape[0]
        scores = np.zeros(n, dtype=np.float64)

        for i in range(n):
            others = np.delete(sim_matrix[i], i)

            if self.similarity_agg == "mean":
                scores[i] = others.mean()
            else:
                scores[i] = np.median(others)

        return scores

    # --------------------------------------------------------
    # Helper: gabungkan bobot similarity dengan proporsi data
    # --------------------------------------------------------

    def _blend_with_data_size(
        self,
        sim_weights: np.ndarray,
        num_examples: List[int],
    ) -> np.ndarray:
        """
        Gabungkan bobot berbasis similarity (sim_weights, hasil softmax,
        sudah > 0 dan sum = 1) dengan proporsi jumlah data tiap client,
        lewat weighted geometric mean di ruang log:

            log(final_i) = (1 - a) * log(sim_weights_i) + a * log(data_share_i)

        dengan a = self.data_influence.

        Kenapa geometric mean (bukan rata-rata biasa)?
        Supaya similarity TETAP jadi faktor dominan/pengali: kalau
        similarity client kecil, bobot akhirnya tetap ditekan turun
        walau datanya banyak -- tidak "dibanjiri" seperti kalau
        digabung pakai penjumlahan/rata-rata aritmatika biasa.

        Kalau data_influence == 0.0, fungsi ini langsung mengembalikan
        sim_weights apa adanya (perilaku lama, tidak ada perubahan).
        """
        if self.data_influence <= 0.0:
            return sim_weights

        num_examples_arr = np.asarray(num_examples, dtype=np.float64)
        data_share = num_examples_arr / num_examples_arr.sum()
        self.last_data_shares = data_share

        eps = 1e-12  # hindari log(0)
        log_sim = np.log(sim_weights + eps)
        log_data = np.log(data_share + eps)

        a = self.data_influence
        log_blended = (1.0 - a) * log_sim + a * log_data

        blended = np.exp(log_blended)
        blended /= blended.sum()

        return blended

    # --------------------------------------------------------
    # Aggregate Fit
    # --------------------------------------------------------

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures,
    ):
        """
        Mengagregasi hasil training dari client menggunakan
        pairwise cosine similarity sebagai dasar bobot.
        """

        # ----------------------------------------------------
        # 1. Validasi hasil client
        # ----------------------------------------------------
        if not results:
            return None, {}
        # ----------------------------------------------------
        # 2. Ambil parameter dari setiap client
        # ----------------------------------------------------
        #
        # Kalau mau pakai feature extraction/embedding alih-alih
        # seluruh parameter model, ganti bagian ini dengan vector
        # embedding yang dikirim client lewat fit_res.metrics.
        # ----------------------------------------------------

        client_thetas: List[np.ndarray] = []
        client_ndarrays: List[List[np.ndarray]] = []

        for _, fit_res in results:
            ndarrays = parameters_to_ndarrays(fit_res.parameters)

            client_ndarrays.append(ndarrays)
            client_thetas.append(_flatten(ndarrays))

        # ----------------------------------------------------
        # 3. Hitung pairwise similarity SEBELUM ada agregasi apapun
        # ----------------------------------------------------
        sim_matrix = self._pairwise_similarity_matrix(client_thetas)
        self.last_similarity_matrix = sim_matrix
        # ----------------------------------------------------
        # 4. Skor representativitas tiap client
        # ----------------------------------------------------
        scores = self._representativeness_scores(sim_matrix)
        # ----------------------------------------------------
        # 5. Konversi skor menjadi aggregation weights (softmax)
        # ----------------------------------------------------

        temperature = max(self.temperature, 1e-6)

        scaled_scores = scores / temperature
        scaled_scores -= scaled_scores.max()

        weights = np.exp(scaled_scores)
        weights /= weights.sum()

        # ----------------------------------------------------
        # 6. Terapkan minimum weight jika diperlukan
        # ----------------------------------------------------

        if self.min_weight > 0.0:
            weights = np.maximum(weights, self.min_weight)
            weights /= weights.sum()

        # ----------------------------------------------------
        # 6b. Gabungkan dengan proporsi jumlah data (opsional)
        # ----------------------------------------------------
        #
        # Similarity tetap prioritas utama -- lihat _blend_with_data_size.
        # Kalau data_influence == 0.0 (default), langkah ini no-op dan
        # weights persis sama seperti versi lama.
        # ----------------------------------------------------

        num_examples = [fit_res.num_examples for _, fit_res in results]
        weights = self._blend_with_data_size(weights, num_examples)

        # ----------------------------------------------------
        # 7. Weighted aggregation setiap layer
        # ----------------------------------------------------
        #
        # Ini baru dilakukan SETELAH bobot dari pairwise
        # similarity didapat -- bukan sebelum, seperti versi lama.
        # ----------------------------------------------------

        num_layers = len(client_ndarrays[0])

        aggregated_ndarrays: List[np.ndarray] = []

        for layer_idx in range(num_layers):

            layer_stack = np.stack(
                [
                    client_ndarrays[client_idx][layer_idx]
                    for client_idx in range(len(client_ndarrays))
                ],
                axis=0,
            )

            weight_shape = [-1] + [1] * (layer_stack.ndim - 1)
            reshaped_weights = weights.reshape(weight_shape)

            aggregated_layer = np.sum(
                layer_stack * reshaped_weights,
                axis=0,
            )

            aggregated_layer = aggregated_layer.astype(
                client_ndarrays[0][layer_idx].dtype
            )

            aggregated_ndarrays.append(aggregated_layer)

        aggregated_parameters = ndarrays_to_parameters(aggregated_ndarrays)

        # ----------------------------------------------------
        # 8. Aggregate metrics dari client
        # ----------------------------------------------------
        aggregated_metrics: Dict[str, Scalar] = {}
        if self.fit_metrics_aggregation_fn is not None:
            fit_metrics = [
                (fit_res.num_examples, fit_res.metrics)
                for _, fit_res in results
            ]

            aggregated_metrics = (
                self.fit_metrics_aggregation_fn(fit_metrics)
            )

        # ----------------------------------------------------
        # 9. Catat skor & bobot tiap client untuk monitoring
        # ----------------------------------------------------

        for idx, (_, fit_res) in enumerate(results):

            client_name = fit_res.metrics.get(
                "client_name",
                f"client_{idx}",
            )

            aggregated_metrics[
                f"pairwise_sim_score_{client_name}"
            ] = float(scores[idx])

            aggregated_metrics[
                f"agg_weight_{client_name}"
            ] = float(weights[idx])

            if self.last_data_shares is not None:
                aggregated_metrics[
                    f"data_share_{client_name}"
                ] = float(self.last_data_shares[idx])

        # ----------------------------------------------------
        # 10. Return hasil agregasi
        # ----------------------------------------------------

        return aggregated_parameters, aggregated_metrics