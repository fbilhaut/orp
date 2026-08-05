use std::collections::HashSet;
use std::path::Path;
use composable::Composable;
use ort::session::{Session, SessionInputs, SessionOutputs, builder::GraphOptimizationLevel};
use crate::error::UnexpectedModelSchemaError;

use super::Result;
use super::params::RuntimeParameters;
use super::pipeline::Pipeline;


/// A `Model` can load an ONNX model, and run it using the provided pipeline.
pub struct Model {    
    session: Session,
}


impl Model {    
    pub fn new<P: AsRef<Path>>(model_path: P, params: RuntimeParameters) -> Result<Self> {
        let session = Session::builder()?
            .with_intra_threads(params.threads())?
            .with_execution_providers(params.into_execution_providers())?
            .with_optimization_level(GraphOptimizationLevel::Level3)?            
            .commit_from_file(model_path)?;

        Ok(Self {
            session,
        })
    }

    pub fn new_from_bytes(model_bytes: &[u8], params: RuntimeParameters) -> Result<Self> {
        let session = Session::builder()?
            .with_intra_threads(params.threads())?
            .with_execution_providers(params.into_execution_providers())?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .commit_from_memory(model_bytes)?;

        Ok(Self {
            session
        })
    }

    /// Perform inferences using the provided pipeline and parameters
    pub fn inference<'a, P: Pipeline<'a>>(&'a self, input: P::Input, pipeline: &P, params: &P::Parameters) -> Result<P::Output> {
        // check schema
        self.check_schema(pipeline, params)?;
        // pre-process
        let (input, context) = pipeline.pre_processor(params).apply(input)?;
        // inference
        let output = self.run(input)?;                
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
        let metadata = self.session.metadata()?;
        writeln!(writer, "NAME: {}", metadata.name()?)?;
        writeln!(writer, "PRODUCER: {}", metadata.producer()?)?;
        writeln!(writer, "VERSION: {}", metadata.version()?)?;
        writeln!(writer, "INPUTS:")?;
        for input in &self.session.inputs {
            writeln!(writer, "\t{}: {:?}", input.name, input.input_type)?;
        }
        writeln!(writer, "OUTPUTS:")?;
        for input in &self.session.outputs {
            writeln!(writer, "\t{}: {:?}", input.name, input.output_type)?;
        }
        Ok(())
    }

    /// Check model schema wrt. pipeline expectations
    fn check_schema<'a, P: Pipeline<'a>>(&'a self, pipeline: &P, params: &P::Parameters) -> Result<()> {
        if let Some(expected_inputs) = pipeline.expected_inputs(params) {
            // inputs should be exactly the same sets
            let expected_inputs = &expected_inputs.collect();
            let actual_inputs: HashSet<_> = self.session.inputs.iter().map(|i| i.name.as_str()).collect();
            if !actual_inputs.eq(expected_inputs) {
                return UnexpectedModelSchemaError::new_for_input(expected_inputs, &actual_inputs).into_err();
            }
        }
        if let Some(expected_outputs) = pipeline.expected_outputs(params) {            
            // for outputs, we just check that the expected ones are present (but having others is ok)
            let expected_outputs = &expected_outputs.collect();
            let actual_outputs: HashSet<_> = self.session.outputs.iter().map(|i| i.name.as_str()).collect();
            if !actual_outputs.is_superset(expected_outputs) {
                return UnexpectedModelSchemaError::new_for_output(expected_outputs, &actual_outputs).into_err();
            }
        }
        Ok(())
    }

    fn run(&self, input: SessionInputs<'_, '_>) -> Result<SessionOutputs<'_, '_>> {
        Ok(self.session.run(input)?)
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