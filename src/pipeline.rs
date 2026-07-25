use ort::session::{SessionInputs, SessionOutputs};
use composable::Composable;
use super::model::Model;
use super::Result;


/// Defines a generic pipeline
pub trait Pipeline<'a> {
    type Input;
    type Output;
    type Context;
    type Parameters;
    
    fn pre_processor(&self, params: &Self::Parameters) -> impl PreProcessor<'a, Self::Input, Self::Context>;
    
    fn post_processor(&self, params: &Self::Parameters) -> impl PostProcessor<'a, Self::Output, Self::Context>;

    fn to_composable(self, model: &'a Model, params: &'a Self::Parameters) -> impl Composable<Self::Input, Self::Output> where Self: Sized {
        ComposablePipeline::new(self, model, params)
    }

    /// Optionally, the pipeline can expose the (exact) set of input tensors that must be exposed by the model 
    /// In such case it will be checked before inferencing.
    fn expected_inputs(&self, _params: &Self::Parameters) -> Option<impl Iterator<Item = &str>> {
        None::<std::iter::Empty<&str>>
    }

    /// Optionally, the pipeline can expose the (sub-)set of output tensors that must be exposed by the model
    /// In such case it will be checked before inferencing.
    fn expected_outputs(&self, _params: &Self::Parameters) -> Option<impl Iterator<Item = &str>> {
        None::<std::iter::Empty<&str>>
    }
}


/// Defines a generic pre-processor
pub trait PreProcessor<'a, I, C>: Composable<I, (SessionInputs<'a, 'a>, C)> {}
impl<'a, I, C, T: Composable<I, (SessionInputs<'a, 'a>, C)>> PreProcessor<'a, I, C> for T {}


/// Defines a generic post-processor
pub trait PostProcessor<'a, O, C>: Composable<(SessionOutputs<'a>, C), O> {}
impl<'a, O, C, T: Composable<(SessionOutputs<'a>, C), O>> PostProcessor<'a, O, C> for T {}


/// Owns a pipeline, and references a model and some parameters to implement `Composable`
struct ComposablePipeline<'a, P: Pipeline<'a>> {
    pipeline: P,
    params: &'a P::Parameters,
    model: &'a Model,        
}


impl<'a, P: Pipeline<'a>> ComposablePipeline<'a, P> {
    pub fn new(pipeline: P, model: &'a Model, params: &'a P::Parameters) -> Self {
        Self { pipeline, params, model }
    }
}


impl<'a, P: Pipeline<'a>> Composable<P::Input, P::Output> for ComposablePipeline<'a, P> {
    fn apply(&self, input: P::Input) -> Result<P::Output> {
        self.model.inference(input, &self.pipeline, self.params)
    }
}
