use std::collections::HashSet;
use std::path::Path;
use std::sync::Mutex;
use composable::Composable;
use ort::session::{Session, SessionOutputs, builder::GraphOptimizationLevel};
use crate::error::UnexpectedModelSchemaError;

use super::Result;
use super::params::RuntimeParameters;
use super::pipeline::Pipeline;


/// Strips the recovery payload from an `ort` builder error so it satisfies
/// `Box<dyn Error + Send + Sync>` (since ort 2.0.0-rc.10, builder methods
/// return `Error<SessionBuilder>` whose payload is neither `Send` nor `Sync`).
fn ort_err<R>(e: ort::Error<R>) -> Box<dyn std::error::Error + Send + Sync> {
    Box::new(ort::Error::new(e.to_string()))
}


/// A `Model` can load an ONNX model, and run it using the provided pipeline.
///
/// Since ort 2.0.0-rc.10, `Session::run` requires `&mut Session`; the session
/// is wrapped in a `Mutex` to preserve this crate's `&self` inference API.
/// The lock is held per `run` call, so concurrent callers serialize on
/// inference (ONNX Runtime parallelizes internally via intra-op threads).
pub struct Model {
    session: Mutex<Session>,
}


impl Model {
    pub fn new<P: AsRef<Path>>(model_path: P, params: RuntimeParameters) -> Result<Self> {
        let session = Session::builder()?
            .with_intra_threads(params.threads()).map_err(ort_err)?
            .with_execution_providers(params.into_execution_providers()).map_err(ort_err)?
            .with_optimization_level(GraphOptimizationLevel::Level3).map_err(ort_err)?
            .commit_from_file(model_path)?;

        Ok(Self {
            session: Mutex::new(session),
        })
    }

    pub fn new_from_bytes(model_bytes: &[u8], params: RuntimeParameters) -> Result<Self> {
        let session = Session::builder()?
            .with_intra_threads(params.threads()).map_err(ort_err)?
            .with_execution_providers(params.into_execution_providers()).map_err(ort_err)?
            .with_optimization_level(GraphOptimizationLevel::Level3).map_err(ort_err)?
            .commit_from_memory(model_bytes)?;

        Ok(Self {
            session: Mutex::new(session),
        })
    }

    /// Perform inferences using the provided pipeline and parameters
    pub fn inference<'a, P: Pipeline<'a>>(&'a self, input: P::Input, pipeline: &P, params: &P::Parameters) -> Result<P::Output> {
        // check schema
        self.check_schema(pipeline, params)?;
        // pre-process
        let (input, context) = pipeline.pre_processor(params).apply(input)?;
        // inference
        let output = {
            let mut session = self.session.lock().map_err(|e| e.to_string())?;
            let output = session.run(input)?;
            // SAFETY: `SessionOutputs<'r>` only borrows the output *names*
            // (`&'r str`) from the session's outlet metadata; the values are
            // owned `DynValue`s. Those name strings live on the heap inside
            // the `Session`, which is owned by `self` (borrowed for `'a`)
            // and is neither dropped nor structurally mutated while `'a` is
            // live. The transmute only widens the guard-scoped borrow of
            // those names to `'a`.
            unsafe { std::mem::transmute::<SessionOutputs<'_>, SessionOutputs<'a>>(output) }
        };
        // post-process
        let output = pipeline.post_processor(params).apply((output, context))?;
        // ok
        Ok(output)
    }

    pub fn to_composable<'a, P: Pipeline<'a>>(&'a self, pipeline: &'a P, params: &'a P::Parameters) -> impl Composable<P::Input, P::Output> {
        ComposableModel::new(self, pipeline, params)
    }

    /// Writes various model properties from metadata and input/output tensors
    pub fn inspect<W: std::io::Write>(&self, mut writer: W) -> Result<()> {
        let session = self.session.lock().map_err(|e| e.to_string())?;
        let metadata = session.metadata()?;
        writeln!(writer, "NAME: {}", metadata.name().unwrap_or_default())?;
        writeln!(writer, "PRODUCER: {}", metadata.producer().unwrap_or_default())?;
        writeln!(writer, "VERSION: {}", metadata.version().unwrap_or_default())?;
        writeln!(writer, "INPUTS:")?;
        for input in session.inputs() {
            writeln!(writer, "\t{}: {:?}", input.name(), input.dtype())?;
        }
        writeln!(writer, "OUTPUTS:")?;
        for input in session.outputs() {
            writeln!(writer, "\t{}: {:?}", input.name(), input.dtype())?;
        }
        Ok(())
    }

    /// Check model schema wrt. pipeline expectations
    fn check_schema<'a, P: Pipeline<'a>>(&'a self, pipeline: &P, params: &P::Parameters) -> Result<()> {
        let session = self.session.lock().map_err(|e| e.to_string())?;
        if let Some(expected_inputs) = pipeline.expected_inputs(params) {
            // inputs should be exactly the same sets
            let expected_inputs = &expected_inputs.collect();
            let actual_inputs: HashSet<_> = session.inputs().iter().map(|i| i.name()).collect();
            if !actual_inputs.eq(expected_inputs) {
                return UnexpectedModelSchemaError::new_for_input(expected_inputs, &actual_inputs).into_err();
            }
        }
        if let Some(expected_outputs) = pipeline.expected_outputs(params) {
            // for outputs, we just check that the expected ones are present (but having others is ok)
            let expected_outputs = &expected_outputs.collect();
            let actual_outputs: HashSet<_> = session.outputs().iter().map(|i| i.name()).collect();
            if !actual_outputs.is_superset(&expected_outputs) {
                return UnexpectedModelSchemaError::new_for_output(expected_outputs, &actual_outputs).into_err();
            }
        }
        Ok(())
    }

}


/// References a model, a pipeline and some parameters to implement `Composable`
struct ComposableModel<'a, P: Pipeline<'a>> {
    model: &'a Model,
    pipeline: &'a P,
    params: &'a P::Parameters,
}


impl<'a, P: Pipeline<'a>> ComposableModel<'a, P> {
    pub fn new(model: &'a Model, pipeline: &'a P, params: &'a P::Parameters) -> Self {
        Self { model, pipeline, params }
    }
}


impl<'a, P: Pipeline<'a>> Composable<P::Input, P::Output> for ComposableModel<'a, P> {
    fn apply(&self, input: P::Input) -> Result<P::Output> {
        self.model.inference(input, self.pipeline, self.params)
    }
}
