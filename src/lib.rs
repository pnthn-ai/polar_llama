pub mod cache;
mod expressions;
pub mod utils;
pub mod model_client;
pub mod ann;
pub mod cost;
pub mod index;
pub mod kmeans;
pub mod mcp;
pub mod metrics;
pub mod quality;
mod stream_pyfn;
mod typesafe_expr;

#[cfg(target_os = "linux")]
use jemallocator::Jemalloc;

#[global_allocator]
#[cfg(target_os = "linux")]
static ALLOC: Jemalloc = Jemalloc;

use pyo3::prelude::*;
use model_client::Provider;
use std::str::FromStr;

// Make PyProvider available at module level for Python
#[pyclass(name = "Provider")]
#[derive(Clone)]
pub struct PyProvider(Provider);

#[pymethods]
impl PyProvider {
    #[classattr]
    const OPENAI: Self = PyProvider(Provider::OpenAI);
    
    #[classattr]
    const ANTHROPIC: Self = PyProvider(Provider::Anthropic);
    
    #[classattr]
    const GEMINI: Self = PyProvider(Provider::Gemini);
    
    #[classattr]
    const GROQ: Self = PyProvider(Provider::Groq);
    
    #[classattr]
    const BEDROCK: Self = PyProvider(Provider::Bedrock);
    
    #[new]
    fn new(provider_str: &str) -> PyResult<Self> {
        match Provider::from_str(provider_str) {
            Ok(provider) => Ok(PyProvider(provider)),
            Err(err) => Err(pyo3::exceptions::PyValueError::new_err(
                format!("Invalid provider: {err}")
            )),
        }
    }
    
    fn __str__(&self) -> String {
        self.0.as_str().to_string()
    }
}


/// List the TypeSafe System One models available to `TYPESAFE_API_KEY`.
///
/// Returns `[{"name", "description", "release_date"}, ...]`. Exposed as a
/// plain function rather than an expression because the catalogue is not
/// row-shaped -- see `polar_llama.typesafe.list_models`.
#[pyfunction]
fn _typesafe_list_models(py: Python<'_>) -> PyResult<Vec<std::collections::HashMap<String, String>>> {
    // Release the GIL: this is a blocking network call.
    let result = py.detach(|| {
        utils::RT.block_on(async {
            model_client::typesafe::list_models(model_client::http_client()).await
        })
    });

    match result {
        Ok(models) => Ok(models
            .into_iter()
            .map(|m| {
                let mut map = std::collections::HashMap::new();
                map.insert("name".to_string(), m.name);
                if let Some(d) = m.description {
                    map.insert("description".to_string(), d);
                }
                if let Some(r) = m.release_date {
                    map.insert("release_date".to_string(), r);
                }
                map
            })
            .collect()),
        Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "TypeSafe list_models failed: {e}"
        ))),
    }
}

// Register expression functions with the Python module
#[pyfunction]
fn register_expressions(_py: Python<'_>) -> PyResult<&'static str> {
    // We don't need to do anything here since the expressions are registered by the polars_expr macro
    // This function exists just to make it explicit in the code that the expressions are registered
    Ok("Expressions registered successfully")
}

#[pymodule]
fn polar_llama(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.setattr("__version__", env!("CARGO_PKG_VERSION"))?;

    // Add the PyProvider class to the module
    m.add_class::<PyProvider>()?;

    // Persistent, incrementally updatable HNSW index (issue #82); wrapped by
    // `polar_llama.index.HnswIndex`.
    m.add_class::<index::PyHnswIndex>()?;

    // Add the register_expressions function to the module
    m.add_function(wrap_pyfunction!(register_expressions, m)?)?;

    // Streaming inference entry point (bypasses the polars_expr/serde kwargs
    // boundary so it can receive a Python callback directly).
    m.add_function(wrap_pyfunction!(stream_pyfn::_stream_inference_batch, m)?)?;

    // TypeSafe System One model catalogue (https://docs.typesafe.ai/api).
    m.add_function(wrap_pyfunction!(_typesafe_list_models, m)?)?;

    Ok(())
}
