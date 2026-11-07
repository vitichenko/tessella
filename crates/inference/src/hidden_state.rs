use candle_core::{DType, Device, Tensor};
pub use tessella_protocol::hidden_state::*;

pub trait PayloadTensorExt {
    /// Serializes a candle Tensor to a Payload.
    ///
    /// # Errors
    ///
    /// Returns `anyhow::Error` if the tensor cannot be read or if the dtype is unsupported.
    fn from_tensor(
        tensor: &Tensor,
        request_id: String,
        seq_start_pos: u32,
        layer_idx: usize,
    ) -> anyhow::Result<Payload>;

    /// Deserializes to a candle Tensor.
    ///
    /// # Errors
    ///
    /// Returns `anyhow::Error` if the conversion fails.
    fn to_tensor(&self, device: &Device) -> anyhow::Result<Tensor>;
}

impl PayloadTensorExt for Payload {
    fn from_tensor(
        tensor: &Tensor,
        request_id: String,
        seq_start_pos: u32,
        layer_idx: usize,
    ) -> anyhow::Result<Self> {
        let shape = tensor.dims().to_vec();
        // Single GPU→CPU transfer before any reads — avoids double CUDA sync.
        let cpu_tensor = tensor.to_device(&candle_core::Device::Cpu)?;
        let flat = cpu_tensor.flatten_all()?;
        let (dtype, data) = match flat.dtype() {
            DType::F32 => (
                DTypeTag::F32,
                bytemuck::cast_slice(&flat.to_vec1::<f32>()?).to_vec(),
            ),
            DType::F16 => (
                DTypeTag::F16,
                bytemuck::cast_slice(&flat.to_vec1::<half::f16>()?).to_vec(),
            ),
            other => anyhow::bail!("Unsupported dtype for hidden state: {:?}", other),
        };

        let checksum = crc32fast::hash(&data);

        Ok(Self {
            request_id,
            session_id: None, // Will be set by caller if needed
            seq_start_pos,
            layer_idx,
            shape,
            dtype,
            data,
            checksum,
            entry_node_url: None,
            signature: Vec::new(),
            client_pubkey: Vec::new(),
            karma_ticket: None,
            work_ticket: None,
            signature_version: tessella_protocol::signing::SIGNATURE_VERSION_LEGACY,
            nonce: 0,
            iat: 0,
            tokens: Vec::new(),
        })
    }

    fn to_tensor(&self, device: &Device) -> anyhow::Result<Tensor> {
        if self.shape.is_empty() || self.data.is_empty() {
            anyhow::bail!(
                "Invalid Payload: shape={:?} data_bytes={}. Payload was not properly initialized.",
                self.shape,
                self.data.len()
            );
        }

        // Integrity check
        if !self.verify_checksum() {
            anyhow::bail!("Checksum mismatch in HiddenStatePayload");
        }

        let bytes_per_elem = match self.dtype {
            DTypeTag::F32 => 4usize,
            DTypeTag::F16 => 2usize,
        };
        let expected_bytes: usize = self.shape.iter().product::<usize>() * bytes_per_elem;
        if self.data.len() != expected_bytes {
            anyhow::bail!(
                "Shape/data mismatch: shape {:?} expects {} bytes, got {}",
                self.shape,
                expected_bytes,
                self.data.len()
            );
        }

        let dtype = match self.dtype {
            DTypeTag::F32 => DType::F32,
            DTypeTag::F16 => DType::F16,
        };
        Ok(Tensor::from_raw_buffer(
            &self.data,
            dtype,
            &self.shape,
            device,
        )?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_roundtrip_f32() {
        let device = Device::Cpu;
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let tensor = Tensor::from_vec(data.clone(), &[1, 4], &device).unwrap();
        let payload = Payload::from_tensor(&tensor, "req-1".into(), 0, 5).unwrap();
        let bytes = payload.to_bytes().unwrap();
        let decoded = Payload::from_bytes(&bytes).unwrap();
        let recovered = decoded.to_tensor(&device).unwrap();
        let recovered_data = recovered.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        assert_eq!(recovered_data, data);
        assert_eq!(decoded.layer_idx, 5);
        assert_eq!(decoded.request_id, "req-1");
        assert_eq!(decoded.seq_start_pos, 0);
    }

    #[test]
    fn test_shape_preserved() {
        let device = Device::Cpu;
        let tensor = Tensor::zeros(&[2, 8, 512], DType::F32, &device).unwrap();
        let payload = Payload::from_tensor(&tensor, "req-2".into(), 10, 10).unwrap();
        assert_eq!(payload.shape, vec![2, 8, 512]);
    }

    #[test]
    fn the_token_count_is_the_sequence_dimension_of_the_wire_shape() {
        let device = Device::Cpu;
        let pipeline = Tensor::zeros(&[1, 2_048, 1_536], DType::F16, &device).unwrap();
        let payload = Payload::from_tensor(&pipeline, "req-3".into(), 0, 7).unwrap();
        assert_eq!(payload.n_tokens(), Some(2_048));
        assert_eq!(payload.shape, vec![1, 2_048, 1_536]);

        let batched = Tensor::zeros(&[2, 8, 512], DType::F32, &device).unwrap();
        assert_eq!(
            Payload::from_tensor(&batched, "req-4".into(), 0, 7)
                .unwrap()
                .n_tokens(),
            Some(8),
            "the sequence is the second dimension even with a batch of two"
        );

        let legacy = Tensor::from_vec(vec![0.0f32; 4], &[4], &device).unwrap();
        assert_eq!(
            Payload::from_tensor(&legacy, "req-5".into(), 0, 7)
                .unwrap()
                .n_tokens(),
            None
        );
    }
}
