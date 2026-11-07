use candle_core::{Device, Tensor};
use tessella_core::error::DomainError;
use tessella_core::models::{ComputeDevice, TensorView};

/// Conversions between Domain models and Candle infrastructure.
pub struct Converter;

impl Converter {
    /// Maps a domain ComputeDevice to a live Candle Device.
    ///
    /// # Errors
    ///
    /// Returns `DomainError` if the device is unsupported or fails to initialize.
    pub fn map_device(device: &ComputeDevice) -> Result<Device, DomainError> {
        match device {
            ComputeDevice::Cpu => Ok(Device::Cpu),
            ComputeDevice::Cuda { device_id } => {
                Device::new_cuda(*device_id).map_err(|e| DomainError::Config(e.to_string()))
            }
            ComputeDevice::Metal => {
                Device::new_metal(0).map_err(|e| DomainError::Config(e.to_string()))
            }
        }
    }

    /// Creates a TensorView from a live Candle Tensor.
    ///
    /// # Errors
    ///
    /// Returns `DomainError` if the tensor cannot be flattened or read.
    pub fn from_tensor(t: &Tensor) -> Result<TensorView, DomainError> {
        let data = t
            .flatten_all()
            .map_err(|e| DomainError::from(crate::error::InferenceError::from(e)))?
            .to_vec1::<f32>()
            .map_err(|e| DomainError::from(crate::error::InferenceError::from(e)))?;
        let shape = t.dims().to_vec();
        Ok(TensorView { data, shape })
    }

    /// Converts back to a live Candle Tensor on the specified device.
    ///
    /// # Errors
    ///
    /// Returns `DomainError` if the conversion fails.
    pub fn to_tensor(view: &TensorView, device: &Device) -> Result<Tensor, DomainError> {
        Tensor::from_vec(view.data.clone(), view.shape.as_slice(), device)
            .map_err(|e| DomainError::from(crate::error::InferenceError::from(e)))
    }
}
