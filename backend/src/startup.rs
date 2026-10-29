//! Composition root: service supervisors outlive HTTP workers, which receive typed proxy handles.

use actix_cors::Cors;
use actix_web::web;

pub(crate) struct Services {
    engine: service::Supervisor,
    data: service::Supervisor,
    gyms: service::Supervisor,
    bench: service::Supervisor,
    health: service::health::Monitor,
}

impl Services {
    /// Start service supervisors and availability monitoring without delaying the API on unavailable siblings.
    pub fn start() -> Self {
        let engine = service::Supervisor::start_background(llm_engine::config());
        let data = service::Supervisor::start_background(data_engine::config());
        let gyms = service::Supervisor::start_background(gyms_engine::config());
        let bench = service::Supervisor::start_background(bench_engine::config());
        let tool_config = tools::Config::from_env();
        let health = service::health::Monitor::start(
            &[
                engine.config().clone(),
                data.config().clone(),
                gyms.config().clone(),
                bench.config().clone(),
            ],
            tool_config.bash_url,
            tool_config.search_url,
            tool_config.search_api_key.is_some(),
        );
        Self {
            engine,
            data,
            gyms,
            bench,
            health,
        }
    }

    /// Wire distinct Actix proxy types from the supervised service configuration without transferring process ownership.
    pub fn application(&self) -> Application {
        let data = data_engine::Proxy::new(
            self.data.base_url(),
            self.data.config().service_secret.clone(),
        );
        let inbound = inbound_policy();
        let search_cache =
            std::sync::Arc::new(api::search_cache::DataEngineCache::new(data.clone()));
        let registry = tools::Registry::from_env(Some(search_cache));
        report_tools(&registry);
        Application {
            data,
            inbound,
            registry,
            health: self.health.subscribe(),
            engine: llm_engine::Proxy::new(
                self.engine.base_url(),
                self.engine.config().service_secret.clone(),
            ),
            gyms: gyms_engine::Proxy::new(
                self.gyms.base_url(),
                self.gyms.config().service_secret.clone(),
            ),
            bench: bench_engine::Proxy::new(
                self.bench.base_url(),
                self.bench.config().service_secret.clone(),
            ),
        }
    }

    /// Request shutdown of all owned children after the HTTP server drains; dropping supervisors remains the fallback.
    pub fn shutdown(&self) {
        self.engine.shutdown();
        self.data.shutdown();
        self.gyms.shutdown();
        self.bench.shutdown();
    }
}

#[derive(Clone)]
pub(crate) struct Application {
    engine: llm_engine::Proxy,
    data: data_engine::Proxy,
    gyms: gyms_engine::Proxy,
    bench: bench_engine::Proxy,
    inbound: service::InboundSecret,
    registry: tools::Registry,
    health: tokio::sync::watch::Receiver<Vec<service::health::Status>>,
}

impl Application {
    /// Register each concrete dependency type for one HTTP worker and mount the unchanged API routes.
    pub fn configure(&self, cfg: &mut web::ServiceConfig) {
        cfg.app_data(web::Data::new(self.engine.clone()))
            .app_data(web::Data::new(self.data.clone()))
            .app_data(web::Data::new(self.gyms.clone()))
            .app_data(web::Data::new(self.bench.clone()))
            .app_data(web::Data::new(self.inbound.clone()))
            .app_data(web::Data::new(self.registry.clone()))
            .app_data(web::Data::new(self.health.clone()))
            .configure(api::configure);
    }
}

/// Construct the established local frontend CORS policy without changing allowed methods or headers.
pub(crate) fn cors() -> Cors {
    Cors::default()
        .allowed_origin("http://localhost:3000")
        .allowed_methods(vec!["GET", "POST", "PUT", "PATCH", "DELETE"])
        .allowed_headers(vec![
            actix_web::http::header::CONTENT_TYPE,
            actix_web::http::header::AUTHORIZATION,
        ])
        .max_age(3600)
}

/// Load reverse-hop authentication and announce the deliberately open development configuration when unset.
fn inbound_policy() -> service::InboundSecret {
    let inbound = service::InboundSecret::from_env();
    if inbound.enforced() {
        println!("inbound service calls authenticated");
    } else {
        println!(
            "inbound service calls are UNAUTHENTICATED — set BACKEND_SERVICE_SECRET here and in the LLM engine to close /internal"
        );
    }
    inbound
}

/// Report configured tool origins or the caveat-only behavior of an empty registry.
fn report_tools(registry: &tools::Registry) {
    match registry.origins() {
        [] => println!("no tools configured — replies that call one get a caveat back"),
        origins => println!("tools ready: {}", origins.join(", ")),
    }
}
